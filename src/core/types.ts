/**
 * Shared type definitions for the VaultSeek core. These types are deliberately
 * free of any Obsidian dependency so the whole `core` layer compiles and unit-tests
 * in plain Node.
 */

/** Identifier of a supported on-device embedding model. */
export type EmbeddingModelId = "Xenova/bge-small-en-v1.5" | "Xenova/all-MiniLM-L6-v2";

/** Q&A generation backend selector. */
export type GenerationBackend = "none" | "ollama" | "lmstudio" | "hosted";

/** Persisted, user-facing configuration. Every key has a default (see config.ts). */
export interface VaultSeekSettings {
  embeddingModel: EmbeddingModelId;
  useWebGPU: boolean;
  chunkTokens: number;
  chunkOverlap: number;
  hybridAlpha: number;
  hnswThreshold: number;
  generationBackend: GenerationBackend;
  ollamaEndpoint: string;
  ollamaModel: string;
  lmstudioEndpoint: string;
  lmstudioModel: string;
  hostedApiKey: string;
  hostedEndpoint: string;
  hostedModel: string;
  excludedFolders: string[];
}

/** A note handed to the indexer, decoupled from Obsidian's `TFile`. */
export interface NoteInput {
  /** Vault-relative path, e.g. `research/caffeine.md`. Stable primary key. */
  path: string;
  /** Display title (frontmatter `title` or basename). */
  title: string;
  /** Raw Markdown body (frontmatter already stripped or retained per caller). */
  content: string;
  /** Last-modified time in epoch ms, used only for change ordering. */
  mtime: number;
}

/** A heading-aware slice of a note, the unit that gets embedded and ranked. */
export interface Chunk {
  /** Deterministic id: `${notePath}#${ordinal}`. */
  id: string;
  notePath: string;
  noteTitle: string;
  /** Nearest enclosing heading text, or empty string for pre-heading content. */
  heading: string;
  /** Zero-based position of this chunk within its note. */
  ordinal: number;
  text: string;
}

/** A chunk plus its content hash, the diffing unit for incremental re-embedding. */
export interface HashedChunk extends Chunk {
  /** Stable hash of `text`; unchanged hash means the chunk need not be re-embedded. */
  hash: string;
}

/** A stored vector entry: chunk metadata alongside its embedding. */
export interface VectorRecord {
  chunk: Chunk;
  hash: string;
  vector: Float32Array;
}

/** A single ranked search hit. */
export interface SearchResult {
  chunk: Chunk;
  /** Final blended score actually used for ordering. */
  score: number;
  /** Raw cosine similarity in [-1, 1] (0 when semantic ranking is disabled). */
  semanticScore: number;
  /** Normalized BM25 score in [0, 1] (0 when lexical ranking is disabled). */
  lexicalScore: number;
  /** Short, query-agnostic preview of the chunk text. */
  snippet: string;
}

/** Which signals to combine when ranking. */
export type RankingMode = "semantic" | "lexical" | "hybrid";

/**
 * Minimal embedder contract. Implementations: a transformers.js model (in a
 * worker or in Node), and a deterministic stub for offline tests.
 */
export interface Embedder {
  /** Output dimensionality of the produced vectors. */
  readonly dim: number;
  /** Identifier recorded in index metadata so a model change forces a re-embed. */
  readonly modelId: string;
  /** Embed a batch of texts into L2-normalized vectors. */
  embed(texts: string[]): Promise<Float32Array[]>;
}

/** Metadata persisted alongside the vector blob to validate a reload. */
export interface IndexMetadata {
  /** Schema version of the on-disk format. */
  version: number;
  /** Model that produced the vectors; a mismatch forces a full re-embed. */
  modelId: string;
  /** Vector dimensionality. */
  dim: number;
  /** Number of vectors in the blob. */
  count: number;
}

/** One message in a chat conversation. */
export interface ChatMessage {
  role: "user" | "assistant";
  /** Message text; assistant messages may contain inline `[[note]]` citations. */
  content: string;
  /** Note paths cited by an assistant message (empty for user messages). */
  citations: string[];
  /** True when an assistant message is a weak-retrieval refusal (offline mode). */
  refused: boolean;
  /**
   * For assistant messages: whether the answer was grounded in a strong note
   * match. When false, the model may have used general knowledge — the UI flags
   * this so the reader knows the answer did not come from their vault.
   */
  grounded?: boolean;
}

/** Result of a cited Q&A request. */
export interface QaResult {
  /** Answer text with inline `[[note]]` citations, or a refusal message. */
  answer: string;
  /** True when retrieval was too weak and the engine declined to answer. */
  refused: boolean;
  /** Note paths cited in the answer, in citation order. */
  citations: string[];
  /** The retrieved chunks used as grounding context. */
  context: SearchResult[];
}

/** Everything a generator needs to produce a grounded answer. */
export interface GenerationRequest {
  /** The user's natural-language question. */
  question: string;
  /** Retrieved, grounding chunks (already above the confidence floor). */
  context: SearchResult[];
  /** Fully-assembled, prompt-injection-safe prompt for LLM backends. */
  prompt: string;
}

/** Pluggable text generator used by the Q&A engine. */
export interface Generator {
  readonly id: GenerationBackend;
  /** Produce an answer grounded in the request's retrieved context. */
  generate(request: GenerationRequest): Promise<string>;
}
