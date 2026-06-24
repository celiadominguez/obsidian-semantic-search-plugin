/**
 * Central configuration for VaultSeek: the default user-facing settings plus
 * every internal tuning constant. Keeping magic numbers here (rather than inline)
 * means the retrieval behaviour is auditable in one place and the layering rule
 * (`core` never imports `obsidian`) is trivially satisfied for settings access.
 */

import type { EmbeddingModelId, GenerationBackend, VaultSeekSettings } from "./types";

/** Per-model retrieval configuration. */
export interface EmbeddingModelInfo {
  /** Output vector dimensionality. */
  dim: number;
  /** Human-readable label for the settings dropdown. */
  label: string;
  /**
   * Instruction prepended to QUERIES (not passages) before embedding. The BGE
   * v1.5 models are asymmetric retrievers trained to expect this prefix on short
   * queries; applying it both improves ranking and widens the gap between
   * genuine queries and out-of-domain noise. Symmetric models use no prefix.
   */
  queryInstruction: string;
  /**
   * Cosine floor for cited Q&A / chat refusal, calibrated per model. Dense
   * embeddings are anisotropic, so the "unrelated" baseline cosine differs by
   * model; below this the engines refuse rather than answer from weak context.
   */
  similarityFloor: number;
}

/** Embedding models the plugin can load on-device. */
export const EMBEDDING_MODELS: Record<EmbeddingModelId, EmbeddingModelInfo> = {
  "Xenova/bge-small-en-v1.5": {
    dim: 384,
    label: "BGE Small EN v1.5 (default)",
    queryInstruction: "Represent this sentence for searching relevant passages: ",
    similarityFloor: 0.5,
  },
  "Xenova/all-MiniLM-L6-v2": {
    // MiniLM is a symmetric model: no query instruction. Its floor is set
    // conservatively and not calibrated against the demo corpus.
    dim: 384,
    label: "all-MiniLM-L6-v2",
    queryInstruction: "",
    similarityFloor: 0.3,
  },
};

/** Default embedding model: small, int8-quantized, 384-dimensional. */
export const DEFAULT_EMBEDDING_MODEL: EmbeddingModelId = "Xenova/bge-small-en-v1.5";

/** Chunking defaults, expressed in approximate whitespace tokens. */
export const DEFAULT_CHUNK_TOKENS = 512;
export const DEFAULT_CHUNK_OVERLAP = 64;

/** Hybrid blend: `score = alpha * cosine + (1 - alpha) * bm25_norm`. */
export const DEFAULT_HYBRID_ALPHA = 0.6;

/** Above this many chunks the vector store switches from exact cosine to HNSW. */
export const DEFAULT_HNSW_THRESHOLD = 20000;

/** BM25 term-frequency saturation (`k1`) and length-normalization (`b`). */
export const BM25_K1 = 1.2;
export const BM25_B = 0.75;

/** HNSW graph construction / query parameters (used only above the threshold). */
export const HNSW_M = 16;
export const HNSW_EF_CONSTRUCTION = 200;
export const HNSW_EF_SEARCH = 64;

/**
 * Cosine-similarity floor for cited Q&A. When the best retrieved chunk scores
 * below this, the answer engine refuses rather than fabricate from weak context.
 */
export const QA_SIMILARITY_FLOOR = 0.35;

/** Number of chunks fed to the Q&A context window. */
export const QA_CONTEXT_CHUNKS = 6;

/** Most recent conversation turns carried into a chat prompt (bounds prompt size). */
export const CHAT_HISTORY_TURNS = 6;

/** Default number of ranked results surfaced by a search. */
export const DEFAULT_TOP_K = 20;

/** Default local Ollama endpoint for opt-in local generation. */
export const DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434";

/** Default Ollama model for local generation. */
export const DEFAULT_OLLAMA_MODEL = "llama3.1:8b";

/** Default LM Studio endpoint (its OpenAI-compatible local server base URL). */
export const DEFAULT_LMSTUDIO_ENDPOINT = "http://localhost:1234/v1";

/** Debounce window (ms) for coalescing vault change events before re-indexing. */
export const INDEX_DEBOUNCE_MS = 1500;

/** Batch size for embedding chunks, balancing throughput against memory. */
export const EMBED_BATCH_SIZE = 32;

/** Available Q&A generation backends. */
export const GENERATION_BACKENDS: GenerationBackend[] = ["none", "ollama", "lmstudio", "hosted"];

/** Factory for the default settings of a fresh, fully-offline install. */
export function defaultSettings(): VaultSeekSettings {
  return {
    embeddingModel: DEFAULT_EMBEDDING_MODEL,
    useWebGPU: true,
    chunkTokens: DEFAULT_CHUNK_TOKENS,
    chunkOverlap: DEFAULT_CHUNK_OVERLAP,
    hybridAlpha: DEFAULT_HYBRID_ALPHA,
    hnswThreshold: DEFAULT_HNSW_THRESHOLD,
    generationBackend: "none",
    ollamaEndpoint: DEFAULT_OLLAMA_ENDPOINT,
    ollamaModel: DEFAULT_OLLAMA_MODEL,
    lmstudioEndpoint: DEFAULT_LMSTUDIO_ENDPOINT,
    lmstudioModel: "",
    hostedApiKey: "",
    hostedEndpoint: "",
    hostedModel: "",
    excludedFolders: [],
  };
}
