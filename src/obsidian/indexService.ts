/**
 * Indexing service: the bridge between Obsidian's vault and the pure `core`.
 *
 * It is the only place that touches `obsidian` for indexing. It reads notes
 * (never writes to them), chunks and embeds changed content incrementally using
 * content hashes, maintains the vector store and BM25 index, and persists the
 * index to the plugin's own data folder. Embedding runs on-device via
 * transformers.js, in batches; indexing is deferred to after layout-ready and
 * debounced on edits so it stays off the interactive path.
 *
 * Read-only guarantee: the only writes this service makes are to
 * `<pluginDir>/index.bin` and `<pluginDir>/index.json`.
 */

import type { App, TFile } from "obsidian";
import { EMBED_BATCH_SIZE } from "../core/config";
import { chunkNote } from "../core/chunker";
import { embedInput, lexicalInput } from "../core/indexSurface";
import { hashText } from "../core/hash";
import { rank } from "../core/hybridRanker";
import { TransformersEmbedder } from "../core/embedder";
import { createGenerator } from "../core/generation";
import { ChatEngine } from "../core/chat";
import { Bm25Index } from "../core/bm25";
import { obsidianHttpClient } from "./obsidianHttp";
import { ortWasmBinary } from "./ortWasm";
import { SIDECAR_VERSION, VectorStore, type VectorSidecar } from "../core/vectorStore";
import { EMBEDDING_MODELS, type EmbeddingModelInfo } from "../core/config";
import type { NoteInput, RankingMode, SearchResult, VaultSleuthSettings } from "../core/types";

const VECTOR_BLOB_FILE = "index.bin";
const SIDECAR_FILE = "index.json";

/** Progress callback fired during a full (re)index. */
export type ProgressCallback = (done: number, total: number) => void;

/** Aggregate index statistics for the UI. */
export interface IndexStats {
  notes: number;
  chunks: number;
  modelId: string;
  usesHnsw: boolean;
}

export class IndexService {
  private readonly app: App;
  private settings: VaultSleuthSettings;
  private readonly pluginDir: string;
  private embedder: TransformersEmbedder;
  private store: VectorStore;
  private bm25 = new Bm25Index();
  private indexedNotes = new Set<string>();
  /** Serializes all index mutations + persistence so they never interleave. */
  private writeChain: Promise<unknown> = Promise.resolve();
  /** Count of index mutations queued or in flight; drives {@link isIndexing}. */
  private activeMutations = 0;

  constructor(app: App, settings: VaultSleuthSettings, pluginDir: string) {
    this.app = app;
    this.settings = settings;
    this.pluginDir = pluginDir;
    this.embedder = this.createEmbedder();
    this.store = this.createStore();
  }

  private modelDim(): number {
    return EMBEDDING_MODELS[this.settings.embeddingModel].dim;
  }

  private createEmbedder(): TransformersEmbedder {
    const localModelPath = this.settings.localModelPath.trim();
    return new TransformersEmbedder({
      modelId: this.settings.embeddingModel,
      dim: this.modelDim(),
      revision: EMBEDDING_MODELS[this.settings.embeddingModel].revision,
      // The ORT WASM engine is inlined into main.js (see ortWasm.ts); handing the
      // bytes to the embedder keeps onnxruntime-web from fetching it from a CDN.
      getWasmBinary: () => Promise.resolve(ortWasmBinary()),
      // Opt-in fully-offline path: load model files from a vault folder instead
      // of downloading them. Only wired up when the user configured a folder.
      readLocalModelFile:
        localModelPath.length > 0
          ? (relativePath) => this.readLocalModelFile(localModelPath, relativePath)
          : undefined,
    });
  }

  /** Read a model file from the user's local model folder, or undefined if absent. */
  private async readLocalModelFile(
    folder: string,
    relativePath: string,
  ): Promise<ArrayBuffer | undefined> {
    try {
      return await this.app.vault.adapter.readBinary(`${folder}/${relativePath}`);
    } catch {
      return undefined;
    }
  }

  private createStore(): VectorStore {
    return new VectorStore({
      dim: this.modelDim(),
      modelId: this.settings.embeddingModel,
      hnswThreshold: this.settings.hnswThreshold,
    });
  }

  /** Apply updated settings; a model change requires a full re-index by the caller. */
  public updateSettings(settings: VaultSleuthSettings): void {
    const modelChanged = settings.embeddingModel !== this.settings.embeddingModel;
    // Switching the model source (download ⇄ local folder) produces the same
    // vectors, so it only needs a fresh embedder, not a re-index.
    const sourceChanged = settings.localModelPath !== this.settings.localModelPath;
    this.settings = settings;
    if (modelChanged || sourceChanged) {
      this.embedder = this.createEmbedder();
    }
    if (modelChanged) {
      this.store = this.createStore();
      this.bm25.clear();
      this.indexedNotes.clear();
    }
  }

  private vectorBlobPath(): string {
    return `${this.pluginDir}/${VECTOR_BLOB_FILE}`;
  }

  private sidecarPath(): string {
    return `${this.pluginDir}/${SIDECAR_FILE}`;
  }

  /** Whether a note path falls under an excluded folder. */
  private isExcluded(path: string): boolean {
    return this.settings.excludedFolders.some(
      (folder) => folder.length > 0 && (path === folder || path.startsWith(`${folder}/`)),
    );
  }

  /** Markdown files eligible for indexing (respecting exclusions). */
  private indexableFiles(): TFile[] {
    return this.app.vault.getMarkdownFiles().filter((file) => !this.isExcluded(file.path));
  }

  private async toNoteInput(file: TFile): Promise<NoteInput> {
    const content = await this.app.vault.cachedRead(file);
    const frontmatter = this.app.metadataCache.getFileCache(file)?.frontmatter;
    const title =
      typeof frontmatter?.title === "string" && frontmatter.title.length > 0
        ? frontmatter.title
        : file.basename;
    return { path: file.path, title, content, mtime: file.stat.mtime };
  }

  /** Load a persisted index, if present and produced by the current model. */
  public async loadPersisted(): Promise<boolean> {
    const adapter = this.app.vault.adapter;
    if (
      !(await adapter.exists(this.sidecarPath())) ||
      !(await adapter.exists(this.vectorBlobPath()))
    ) {
      return false;
    }
    try {
      const sidecar = JSON.parse(await adapter.read(this.sidecarPath())) as VectorSidecar;
      // Reject an index produced by a different model, dimension, or on-disk
      // schema version — any mismatch means the vectors can't be trusted, so we
      // fall through to a fresh re-index rather than load stale/incompatible data.
      if (
        sidecar.meta.modelId !== this.settings.embeddingModel ||
        sidecar.meta.dim !== this.modelDim() ||
        sidecar.meta.version !== SIDECAR_VERSION
      ) {
        return false;
      }
      const buffer = await adapter.readBinary(this.vectorBlobPath());
      this.store = VectorStore.fromBlob(buffer, sidecar, this.settings.hnswThreshold);
      this.bm25.clear();
      this.indexedNotes.clear();
      for (const entry of sidecar.entries) {
        this.bm25.add(entry.chunk.id, lexicalInput(entry.chunk));
        this.indexedNotes.add(entry.chunk.notePath);
      }
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Run an index mutation exclusively: it starts only after any in-flight
   * mutation finishes, so re-index, incremental edits, deletes, and renames
   * never interleave on the shared store/BM25 (and persistence stays atomic).
   */
  private runExclusive<T>(op: () => Promise<T>): Promise<T> {
    this.activeMutations++;
    const result = this.writeChain.then(op, op);
    this.writeChain = result.then(
      () => undefined,
      () => undefined,
    );
    const settle = (): void => {
      this.activeMutations--;
    };
    // Decrement on settle without rethrowing, so the caller still sees the
    // original result (and its rejection) but this bookkeeping never leaks one.
    result.then(settle, settle);
    return result;
  }

  /**
   * Whether a (re)index is queued or running. Search/chat still work while true,
   * but over a not-yet-complete index — the UI uses this to warn that results may
   * be incomplete until indexing finishes.
   */
  public get isIndexing(): boolean {
    return this.activeMutations > 0;
  }

  /** Persist the current index (vectors blob + JSON sidecar). */
  private async persistIndex(): Promise<void> {
    const adapter = this.app.vault.adapter;
    if (!(await adapter.exists(this.pluginDir))) {
      await adapter.mkdir(this.pluginDir);
    }
    const { vectors, sidecar } = this.store.toBlob();
    await adapter.writeBinary(this.vectorBlobPath(), vectors);
    await adapter.write(this.sidecarPath(), JSON.stringify(sidecar));
  }

  /** Incrementally (re)index one note into the given store/BM25/set by hash diff. */
  private async indexNoteInto(
    note: NoteInput,
    store: VectorStore,
    bm25: Bm25Index,
    indexed: Set<string>,
  ): Promise<void> {
    const chunks = chunkNote(note, this.settings.chunkTokens, this.settings.chunkOverlap);
    const hashes = new Map(chunks.map((chunk) => [chunk.id, hashText(chunk.text)]));
    const existing = store.hashesForNote(note.path);

    // Remove chunks that no longer exist in the note.
    for (const id of existing.keys()) {
      if (!hashes.has(id)) {
        store.remove(id);
        bm25.remove(id);
      }
    }

    // Re-embed only chunks whose content hash changed (or are new).
    const changed = chunks.filter((chunk) => existing.get(chunk.id) !== hashes.get(chunk.id));
    for (let i = 0; i < changed.length; i += EMBED_BATCH_SIZE) {
      const batch = changed.slice(i, i + EMBED_BATCH_SIZE);
      const vectors = await this.embedder.embed(batch.map(embedInput));
      batch.forEach((chunk, j) => {
        store.upsert({ chunk, hash: hashes.get(chunk.id) as string, vector: vectors[j] });
        bm25.add(chunk.id, lexicalInput(chunk));
      });
    }

    if (chunks.length > 0) {
      indexed.add(note.path);
    } else {
      indexed.delete(note.path);
    }
  }

  /** Drop a note's chunks from the given store/BM25/set. */
  private removeNoteFrom(
    path: string,
    store: VectorStore,
    bm25: Bm25Index,
    indexed: Set<string>,
  ): void {
    for (const id of store.hashesForNote(path).keys()) {
      bm25.remove(id);
    }
    store.removeNote(path);
    indexed.delete(path);
  }

  /** Incrementally index a batch of changed files, then persist (serialized). */
  public indexFiles(files: TFile[]): Promise<void> {
    return this.runExclusive(async () => {
      for (const file of files) {
        if (this.isExcluded(file.path)) {
          continue;
        }
        await this.indexNoteInto(
          await this.toNoteInput(file),
          this.store,
          this.bm25,
          this.indexedNotes,
        );
      }
      await this.persistIndex();
    });
  }

  /** Drop a note's chunks and persist (serialized; for delete events). */
  public removeNote(path: string): Promise<void> {
    return this.runExclusive(async () => {
      this.removeNoteFrom(path, this.store, this.bm25, this.indexedNotes);
      await this.persistIndex();
    });
  }

  /** Handle a rename: drop the old path, index the new file, persist (serialized). */
  public renameNote(oldPath: string, file: TFile): Promise<void> {
    return this.runExclusive(async () => {
      this.removeNoteFrom(oldPath, this.store, this.bm25, this.indexedNotes);
      if (!this.isExcluded(file.path)) {
        await this.indexNoteInto(
          await this.toNoteInput(file),
          this.store,
          this.bm25,
          this.indexedNotes,
        );
      }
      await this.persistIndex();
    });
  }

  /**
   * Full (re)index into a fresh index that is swapped in atomically once built,
   * then persisted — all serialized. A concurrent search therefore sees either
   * the previous complete index or the new one, never a half-cleared store.
   */
  public reindexAll(onProgress?: ProgressCallback): Promise<void> {
    return this.runExclusive(async () => {
      const store = this.createStore();
      const bm25 = new Bm25Index();
      const indexed = new Set<string>();
      const files = this.indexableFiles();
      for (let i = 0; i < files.length; i++) {
        await this.indexNoteInto(await this.toNoteInput(files[i]), store, bm25, indexed);
        onProgress?.(i + 1, files.length);
      }
      this.store = store;
      this.bm25 = bm25;
      this.indexedNotes = indexed;
      await this.persistIndex();
    });
  }

  /** Per-model retrieval configuration (query instruction + refusal floor). */
  private modelInfo(): EmbeddingModelInfo {
    return EMBEDDING_MODELS[this.settings.embeddingModel];
  }

  /** Run a ranked search in the requested mode. */
  public async search(query: string, mode: RankingMode, topK: number): Promise<SearchResult[]> {
    return rank({
      query,
      embedder: this.embedder,
      store: this.store,
      bm25: this.bm25,
      alpha: this.settings.hybridAlpha,
      mode,
      topK,
      queryInstruction: this.modelInfo().queryInstruction,
    });
  }

  /** Create a fresh multi-turn chat engine bound to the current index and settings. */
  public createChatEngine(): ChatEngine {
    return new ChatEngine({
      embedder: this.embedder,
      store: this.store,
      bm25: this.bm25,
      generator: createGenerator(this.settings, obsidianHttpClient),
      alpha: this.settings.hybridAlpha,
      similarityFloor: this.modelInfo().similarityFloor,
      queryInstruction: this.modelInfo().queryInstruction,
    });
  }

  /**
   * Whether a usable generative backend is configured. Beyond "not none", the
   * selected backend must actually have the fields it needs, so chat isn't
   * enabled into an opaque failure (e.g. hosted with no endpoint/model/key).
   */
  public get hasGenerativeBackend(): boolean {
    const s = this.settings;
    switch (s.generationBackend) {
      case "ollama":
        return s.ollamaEndpoint.length > 0 && s.ollamaModel.length > 0;
      case "lmstudio":
        return s.lmstudioEndpoint.length > 0 && s.lmstudioModel.length > 0;
      case "hosted":
        return s.hostedEndpoint.length > 0 && s.hostedModel.length > 0 && s.hostedApiKey.length > 0;
      default:
        return false;
    }
  }

  /** Human-readable summary of the active answer model, for the chat header. */
  public get generationSummary(): string {
    const s = this.settings;
    switch (s.generationBackend) {
      case "ollama":
        return `Ollama · ${s.ollamaModel || "no model set"}`;
      case "lmstudio":
        return `LM Studio · ${s.lmstudioModel || "no model set"}`;
      case "hosted":
        return `Hosted · ${s.hostedModel || "no model set"}`;
      case "none":
      default:
        return "Retrieval-only (offline, no model)";
    }
  }

  /** Current index statistics. */
  public stats(): IndexStats {
    return {
      notes: this.indexedNotes.size,
      chunks: this.store.size,
      modelId: this.settings.embeddingModel,
      usesHnsw: this.store.usesHnsw,
    };
  }
}
