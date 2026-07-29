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

import { normalizePath, type App, type TFile } from "obsidian";
import { EMBED_BATCH_SIZE, QA_SIMILARITY_FLOOR } from "../core/config";
import { RemoteEmbedder } from "../core/remoteEmbedder";
import { chunkNote } from "../core/chunker";
import { embedInput, lexicalInput } from "../core/indexSurface";
import { hashText } from "../core/hash";
import { rank } from "../core/hybridRanker";
import { TransformersEmbedder } from "../core/embedder";
import { createGenerator, listOllamaModels, listOpenAiModels } from "../core/generation";
import { ChatEngine } from "../core/chat";
import { Bm25Index } from "../core/bm25";
import { obsidianHttpClient } from "./obsidianHttp";
import { obsidianFetch } from "./obsidianFetch";
import { ortWasmBinary } from "./ortWasm";
import { SIDECAR_VERSION, VectorStore, type VectorSidecar } from "../core/vectorStore";
import { EMBEDDING_MODELS, type EmbeddingModelInfo } from "../core/config";
import type {
  Embedder,
  NoteInput,
  RankingMode,
  SearchResult,
  VaultSleuthSettings,
} from "../core/types";

const VECTOR_BLOB_FILE = "index.bin";
const SIDECAR_FILE = "index.json";

// On-device embedding is CPU-bound and runs on the main thread. To keep the UI
// responsive during a long index (so the app can paint and accept input, e.g.
// closing the settings window), the indexing loops hand the event loop a turn
// whenever more than this many milliseconds of work have elapsed since the last.
const UI_YIELD_INTERVAL_MS = 50;

/** Progress callback fired during a full (re)index. */
export type ProgressCallback = (done: number, total: number) => void;

/** Aggregate index statistics for the UI. */
export interface IndexStats {
  notes: number;
  chunks: number;
  modelId: string;
  usesHnsw: boolean;
}

/** Outcome of probing whether the configured chat backend can actually answer. */
export interface GenerationReadiness {
  ok: boolean;
  /** Present only when `ok` is false: a user-facing explanation of what to fix. */
  reason?: string;
}

export class IndexService {
  private readonly app: App;
  private settings: VaultSleuthSettings;
  private readonly pluginDir: string;
  private embedder: Embedder;
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
    this.pluginDir = normalizePath(pluginDir);
    this.embedder = this.createEmbedder();
    this.store = this.createStore();
  }

  private modelDim(): number {
    if (this.settings.embeddingSource === "remote") {
      return this.settings.remoteEmbeddingDim;
    }
    return EMBEDDING_MODELS[this.settings.embeddingModel].dim;
  }

  /**
   * Build the configured embedder: the bundled on-device model by default, or a
   * remote embeddings server when the user opts in.
   */
  private createEmbedder(): Embedder {
    if (this.settings.embeddingSource === "remote") {
      return new RemoteEmbedder({
        protocol: this.settings.remoteEmbeddingProtocol,
        endpoint: this.settings.remoteEmbeddingEndpoint,
        model: this.settings.remoteEmbeddingModel,
        dim: this.settings.remoteEmbeddingDim,
        apiKey: this.settings.remoteEmbeddingApiKey,
        http: obsidianHttpClient,
      });
    }
    const localModelPath = this.settings.localModelPath.trim();
    return new TransformersEmbedder({
      modelId: this.settings.embeddingModel,
      dim: this.modelDim(),
      revision: EMBEDDING_MODELS[this.settings.embeddingModel].revision,
      // The ORT WASM engine is inlined into main.js (see ortWasm.ts); handing the
      // bytes to the embedder keeps onnxruntime-web from fetching it from a CDN.
      getWasmBinary: () => Promise.resolve(ortWasmBinary()),
      // Download model files via requestUrl — a direct fetch to huggingface.co
      // from the renderer is blocked by Obsidian's CSP/CORS.
      fetchImpl: obsidianFetch,
      // Opt-in fully-offline path: load model files from a vault folder instead
      // of downloading them. Only wired up when the user configured a folder.
      readLocalModelFile:
        localModelPath.length > 0
          ? (relativePath) => this.readLocalModelFile(localModelPath, relativePath)
          : undefined,
    });
  }

  /**
   * Read a model file from the user's local model folder via the Vault API
   * (the files live inside the vault), or undefined if absent. The path is
   * normalized since the folder comes from user-entered settings.
   */
  private async readLocalModelFile(
    folder: string,
    relativePath: string,
  ): Promise<ArrayBuffer | undefined> {
    const file = this.app.vault.getFileByPath(normalizePath(`${folder}/${relativePath}`));
    if (file === null) {
      return undefined;
    }
    try {
      return await this.app.vault.readBinary(file);
    } catch {
      return undefined;
    }
  }

  private createStore(): VectorStore {
    return new VectorStore({
      dim: this.modelDim(),
      // Use the embedder's own id so a remote server/model swap is recorded in
      // the sidecar and invalidates the persisted vectors just like a local one.
      modelId: this.embedder.modelId,
      hnswThreshold: this.settings.hnswThreshold,
    });
  }

  /**
   * Identity of the vectors a settings object produces. Any change here means
   * the persisted embeddings are no longer comparable and must be rebuilt.
   */
  private static vectorIdentity(s: VaultSleuthSettings): string {
    return s.embeddingSource === "remote"
      ? [
          "remote",
          s.remoteEmbeddingProtocol,
          s.remoteEmbeddingEndpoint,
          s.remoteEmbeddingModel,
          s.remoteEmbeddingDim,
        ].join(":")
      : `on-device:${s.embeddingModel}`;
  }

  /** Apply updated settings; a model change requires a full re-index by the caller. */
  public updateSettings(settings: VaultSleuthSettings): void {
    const modelChanged =
      IndexService.vectorIdentity(settings) !== IndexService.vectorIdentity(this.settings);
    // Switching the model source (download ⇄ local folder) or the remote API key
    // produces the same vectors, so it only needs a fresh embedder, not a re-index.
    const sourceChanged =
      settings.localModelPath !== this.settings.localModelPath ||
      settings.remoteEmbeddingApiKey !== this.settings.remoteEmbeddingApiKey;
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
    return normalizePath(`${this.pluginDir}/${VECTOR_BLOB_FILE}`);
  }

  private sidecarPath(): string {
    return normalizePath(`${this.pluginDir}/${SIDECAR_FILE}`);
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
        sidecar.meta.modelId !== this.embedder.modelId ||
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

  /**
   * Fail fast with an actionable message when the remote embedding source is
   * selected but not fully configured, instead of surfacing a confusing
   * dimension mismatch from deep inside an indexing run.
   */
  private assertEmbedderConfigured(): void {
    if (this.settings.embeddingSource !== "remote") {
      return;
    }
    if (this.settings.remoteEmbeddingModel.length === 0) {
      throw new Error("No remote embedding model selected — choose one in settings.");
    }
    if (this.settings.remoteEmbeddingDim <= 0) {
      throw new Error(
        "Remote embedding dimension is not set — use Detect in settings to read it from the server.",
      );
    }
  }

  /**
   * A cooperative yielder for the hot indexing loops. Call the returned function
   * after each note; it hands the event loop a turn (via a macrotask, which — unlike
   * a microtask — lets the browser paint and process input) whenever more than
   * {@link UI_YIELD_INTERVAL_MS} of synchronous work has elapsed. Time-sliced so the
   * added overhead stays negligible regardless of vault size.
   */
  private static makeUiYielder(): () => Promise<void> {
    let last = performance.now();
    return async () => {
      if (performance.now() - last > UI_YIELD_INTERVAL_MS) {
        await new Promise<void>((resolve) => window.setTimeout(resolve));
        last = performance.now();
      }
    };
  }

  /** Incrementally index a batch of changed files, then persist (serialized). */
  public indexFiles(files: TFile[]): Promise<void> {
    return this.runExclusive(async () => {
      this.assertEmbedderConfigured();
      const yieldToUi = IndexService.makeUiYielder();
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
        await yieldToUi();
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
      this.assertEmbedderConfigured();
      const store = this.createStore();
      const bm25 = new Bm25Index();
      const indexed = new Set<string>();
      const files = this.indexableFiles();
      const yieldToUi = IndexService.makeUiYielder();
      for (let i = 0; i < files.length; i++) {
        await this.indexNoteInto(await this.toNoteInput(files[i]), store, bm25, indexed);
        onProgress?.(i + 1, files.length);
        await yieldToUi();
      }
      this.store = store;
      this.bm25 = bm25;
      this.indexedNotes = indexed;
      await this.persistIndex();
    });
  }

  /**
   * Per-model retrieval configuration (query instruction + refusal floor).
   *
   * A remote model is arbitrary, so it uses the shared conservative refusal floor
   * and the user-supplied query instruction (empty unless they set one for an
   * asymmetric retriever like BGE).
   */
  private modelInfo(): EmbeddingModelInfo {
    if (this.settings.embeddingSource === "remote") {
      return {
        dim: this.settings.remoteEmbeddingDim,
        label: this.settings.remoteEmbeddingModel,
        queryInstruction: this.settings.remoteEmbeddingQueryInstruction,
        similarityFloor: QA_SIMILARITY_FLOOR,
        revision: "",
      };
    }
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

  /**
   * Probe whether the configured local generation backend is actually ready to
   * answer: reachable and serving the selected model. Only local servers (Ollama
   * / LM Studio) are probed — the check reuses their model-listing endpoints. The
   * `hosted` and `none` backends return ready without a network call, since we
   * never contact a hosted endpoint except to answer a real question.
   */
  public async checkGenerationReadiness(): Promise<GenerationReadiness> {
    const s = this.settings;
    const probe = async (
      label: string,
      endpoint: string,
      model: string,
      list: () => Promise<string[]>,
    ): Promise<GenerationReadiness> => {
      if (model.length === 0) {
        return {
          ok: false,
          reason: `no ${label} model is set — choose a loaded model in settings.`,
        };
      }
      let models: string[];
      try {
        models = await list();
      } catch {
        return {
          ok: false,
          reason: `${label} isn't reachable at ${endpoint} — start its local server, then try again.`,
        };
      }
      if (!models.includes(model)) {
        const loaded = models.length > 0 ? ` Loaded now: ${models.join(", ")}.` : "";
        return { ok: false, reason: `${label} model "${model}" isn't loaded.${loaded}` };
      }
      return { ok: true };
    };

    switch (s.generationBackend) {
      case "ollama":
        return probe("Ollama", s.ollamaEndpoint, s.ollamaModel, () =>
          listOllamaModels(s.ollamaEndpoint, obsidianHttpClient),
        );
      case "lmstudio":
        return probe("LM Studio", s.lmstudioEndpoint, s.lmstudioModel, () =>
          listOpenAiModels(s.lmstudioEndpoint, undefined, obsidianHttpClient),
        );
      default:
        return { ok: true };
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
      modelId: this.embedder.modelId,
      usesHnsw: this.store.usesHnsw,
    };
  }
}
