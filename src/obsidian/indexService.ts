/**
 * Indexing service: the bridge between Obsidian's vault and the pure `core`.
 *
 * It is the only place that touches `obsidian` for indexing. It reads notes
 * (never writes to them), chunks and embeds changed content incrementally using
 * content hashes, maintains the vector store and BM25 index, and persists the
 * index to the plugin's own data folder. Embedding runs in the Web Worker via
 * `WorkerEmbedder`, so the UI thread stays responsive.
 *
 * Read-only guarantee: the only writes this service makes are to
 * `<pluginDir>/index.bin` and `<pluginDir>/index.json`.
 */

import type { App, TFile } from "obsidian";
import { EMBED_BATCH_SIZE } from "../core/config";
import { chunkNote } from "../core/chunker";
import { hashText } from "../core/hash";
import { rank } from "../core/hybridRanker";
import { createGenerator } from "../core/qa";
import { ChatEngine } from "../core/chat";
import { Bm25Index } from "../core/bm25";
import { obsidianHttpClient } from "./obsidianHttp";
import { VectorStore, type VectorSidecar } from "../core/vectorStore";
import { WorkerEmbedder } from "../worker/workerEmbedder";
import { EMBEDDING_MODELS, type EmbeddingModelInfo } from "../core/config";
import type { Chunk, NoteInput, RankingMode, SearchResult, VaultSeekSettings } from "../core/types";

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

/** Combine title and heading context with chunk text for lexical indexing. */
function lexicalInput(chunk: Chunk): string {
  return `${chunk.noteTitle} ${chunk.heading} ${chunk.text}`.trim();
}

/** Build the text actually fed to the embedding model for a chunk. */
function embedInput(chunk: Chunk): string {
  return chunk.heading.length > 0 ? `${chunk.heading}\n${chunk.text}` : chunk.text;
}

export class IndexService {
  private readonly app: App;
  private settings: VaultSeekSettings;
  private readonly pluginDir: string;
  private embedder: WorkerEmbedder;
  private store: VectorStore;
  private readonly bm25 = new Bm25Index();
  private readonly indexedNotes = new Set<string>();

  constructor(app: App, settings: VaultSeekSettings, pluginDir: string) {
    this.app = app;
    this.settings = settings;
    this.pluginDir = pluginDir;
    this.embedder = this.createEmbedder();
    this.store = this.createStore();
  }

  private modelDim(): number {
    return EMBEDDING_MODELS[this.settings.embeddingModel].dim;
  }

  private createEmbedder(): WorkerEmbedder {
    return new WorkerEmbedder({
      modelId: this.settings.embeddingModel,
      dim: this.modelDim(),
      useWebGPU: this.settings.useWebGPU,
    });
  }

  private createStore(): VectorStore {
    return new VectorStore({
      dim: this.modelDim(),
      modelId: this.settings.embeddingModel,
      hnswThreshold: this.settings.hnswThreshold,
    });
  }

  /** Apply updated settings; a model change requires a full re-index by the caller. */
  public updateSettings(settings: VaultSeekSettings): void {
    const modelChanged = settings.embeddingModel !== this.settings.embeddingModel;
    this.settings = settings;
    if (modelChanged) {
      this.embedder.terminate();
      this.embedder = this.createEmbedder();
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
      if (sidecar.meta.modelId !== this.settings.embeddingModel) {
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

  /** Persist the current index (vectors blob + JSON sidecar). */
  public async persist(): Promise<void> {
    const adapter = this.app.vault.adapter;
    if (!(await adapter.exists(this.pluginDir))) {
      await adapter.mkdir(this.pluginDir);
    }
    const { vectors, sidecar } = this.store.toBlob();
    await adapter.writeBinary(this.vectorBlobPath(), vectors);
    await adapter.write(this.sidecarPath(), JSON.stringify(sidecar));
  }

  /** Index (or re-index) a single note incrementally using content hashes. */
  private async indexNote(note: NoteInput): Promise<void> {
    const chunks = chunkNote(note, this.settings.chunkTokens, this.settings.chunkOverlap);
    const hashes = new Map(chunks.map((chunk) => [chunk.id, hashText(chunk.text)]));
    const existing = this.store.hashesForNote(note.path);

    // Remove chunks that no longer exist in the note.
    for (const id of existing.keys()) {
      if (!hashes.has(id)) {
        this.store.remove(id);
        this.bm25.remove(id);
      }
    }

    // Re-embed only chunks whose content hash changed (or are new).
    const changed = chunks.filter((chunk) => existing.get(chunk.id) !== hashes.get(chunk.id));
    for (let i = 0; i < changed.length; i += EMBED_BATCH_SIZE) {
      const batch = changed.slice(i, i + EMBED_BATCH_SIZE);
      const vectors = await this.embedder.embed(batch.map(embedInput));
      batch.forEach((chunk, j) => {
        this.store.upsert({ chunk, hash: hashes.get(chunk.id) as string, vector: vectors[j] });
        this.bm25.add(chunk.id, lexicalInput(chunk));
      });
    }

    if (chunks.length > 0) {
      this.indexedNotes.add(note.path);
    } else {
      this.indexedNotes.delete(note.path);
    }
  }

  /** Index a file by path (used by vault change events). */
  public async indexFile(file: TFile): Promise<void> {
    if (this.isExcluded(file.path)) {
      return;
    }
    await this.indexNote(await this.toNoteInput(file));
  }

  /** Drop a note's chunks from the index (delete event). */
  public removeNote(path: string): void {
    for (const id of this.store.hashesForNote(path).keys()) {
      this.bm25.remove(id);
    }
    this.store.removeNote(path);
    this.indexedNotes.delete(path);
  }

  /** Handle a rename: drop the old path and index the new file. */
  public async renameNote(oldPath: string, file: TFile): Promise<void> {
    this.removeNote(oldPath);
    await this.indexFile(file);
  }

  /** Full (re)index of every eligible note, reporting progress. */
  public async reindexAll(onProgress?: ProgressCallback): Promise<void> {
    this.store.clear();
    this.bm25.clear();
    this.indexedNotes.clear();
    const files = this.indexableFiles();
    for (let i = 0; i < files.length; i++) {
      await this.indexFile(files[i]);
      onProgress?.(i + 1, files.length);
    }
    await this.persist();
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

  /** Whether a generative backend is configured (chat replies synthesize vs extract). */
  public get hasGenerativeBackend(): boolean {
    return this.settings.generationBackend !== "none";
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

  /** Release the worker on plugin unload. */
  public dispose(): void {
    this.embedder.terminate();
  }
}
