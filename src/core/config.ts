/**
 * Central configuration for VaultSeek: the default user-facing settings plus
 * every internal tuning constant. Keeping magic numbers here (rather than inline)
 * means the retrieval behaviour is auditable in one place and the layering rule
 * (`core` never imports `obsidian`) is trivially satisfied for settings access.
 */

import type { EmbeddingModelId, GenerationBackend, VaultSeekSettings } from "./types";

/** Embedding models the plugin can load on-device, with their output dimensions. */
export const EMBEDDING_MODELS: Record<EmbeddingModelId, { dim: number; label: string }> = {
  "Xenova/bge-small-en-v1.5": { dim: 384, label: "BGE Small EN v1.5 (default)" },
  "Xenova/all-MiniLM-L6-v2": { dim: 384, label: "all-MiniLM-L6-v2" },
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

/** Default number of ranked results surfaced by a search. */
export const DEFAULT_TOP_K = 20;

/** Default local Ollama endpoint for opt-in local generation. */
export const DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434";

/** Default Ollama model for local generation. */
export const DEFAULT_OLLAMA_MODEL = "llama3.1:8b";

/** Debounce window (ms) for coalescing vault change events before re-indexing. */
export const INDEX_DEBOUNCE_MS = 1500;

/** Batch size for embedding chunks, balancing throughput against memory. */
export const EMBED_BATCH_SIZE = 32;

/** Available Q&A generation backends. */
export const GENERATION_BACKENDS: GenerationBackend[] = ["none", "ollama", "hosted"];

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
    hostedApiKey: "",
    hostedEndpoint: "",
    hostedModel: "",
    excludedFolders: [],
  };
}
