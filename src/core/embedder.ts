/**
 * Embedding utilities and implementations.
 *
 * Two concerns live here: pure vector math (mean-pooling, L2-normalization,
 * cosine) shared by every code path, and two `Embedder` implementations:
 *
 *  - `TransformersEmbedder` — the real on-device model via transformers.js. It
 *    runs unchanged in Node (for the offline eval) and in the browser/Web Worker
 *    (for the plugin), selecting WebGPU when available and falling back to WASM.
 *  - `HashingEmbedder` — a deterministic, network-free feature-hashing embedder
 *    used by unit/acceptance tests so the suite never downloads a model. Its
 *    cosine correlates with lexical overlap, which is enough for tests that only
 *    assert determinism, round-tripping, and plausible ranking.
 *
 * Nothing here imports `obsidian`; the heavy transformers.js dependency is
 * loaded lazily so test paths that use only `HashingEmbedder` stay light.
 */

import { hashText } from "./hash";
import type { Embedder, EmbeddingModelId } from "./types";

/** Mean-pool a [tokens, dim] matrix down to a single [dim] vector. */
export function meanPool(tokenVectors: Float32Array[], dim: number): Float32Array {
  const pooled = new Float32Array(dim);
  if (tokenVectors.length === 0) {
    return pooled;
  }
  for (const vec of tokenVectors) {
    for (let i = 0; i < dim; i++) {
      pooled[i] += vec[i];
    }
  }
  const inv = 1 / tokenVectors.length;
  for (let i = 0; i < dim; i++) {
    pooled[i] *= inv;
  }
  return pooled;
}

/** L2-normalize a vector in place and return it. Zero vectors are left as-is. */
export function l2Normalize(vector: Float32Array): Float32Array {
  let sumSquares = 0;
  for (let i = 0; i < vector.length; i++) {
    sumSquares += vector[i] * vector[i];
  }
  const norm = Math.sqrt(sumSquares);
  if (norm > 0) {
    for (let i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }
  return vector;
}

/**
 * Cosine similarity of two equal-length vectors. When both inputs are already
 * L2-normalized (as every embedder here guarantees), this is just the dot
 * product, but the explicit normalization keeps it correct for any input.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Quantization dtype accepted by transformers.js (int8 keeps the model ~33 MB). */
type Dtype = "q8" | "int8" | "uint8" | "fp32" | "fp16" | "q4" | "auto";

/** Execution backend accepted by transformers.js. */
type Device = "webgpu" | "wasm" | "cpu" | "coreml" | "auto";

/**
 * Non-GPU device for the current runtime: transformers.js exposes "wasm" in the
 * browser/Web Worker but "cpu" under Node, so the fallback device differs by
 * environment. This keeps the same model working in the plugin and in the eval.
 */
function nonGpuDevice(): Device {
  const isNode =
    typeof process !== "undefined" &&
    Boolean(process.versions) &&
    typeof process.versions.node === "string";
  return isNode ? "cpu" : "wasm";
}

/** Default quantization: int8, keeping the bge-small model around 33 MB. */
const DEFAULT_DTYPE: Dtype = "q8";

interface TransformersEmbedderOptions {
  modelId: EmbeddingModelId;
  dim: number;
  /** Prefer WebGPU; falls back to WASM automatically when unavailable. */
  useWebGPU?: boolean;
  dtype?: Dtype;
}

// Minimal structural type for the transformers.js feature-extraction pipeline,
// avoiding a hard compile-time dependency on the library's exported types.
type FeatureExtractionPipeline = (
  texts: string[],
  options: { pooling: "mean"; normalize: boolean },
) => Promise<{ tolist(): number[][] }>;

/**
 * On-device embedder backed by transformers.js feature-extraction. The model is
 * fetched once (then cached on disk) and produces mean-pooled, L2-normalized
 * sentence embeddings.
 */
export class TransformersEmbedder implements Embedder {
  public readonly dim: number;
  public readonly modelId: string;
  private readonly useWebGPU: boolean;
  private readonly dtype: Dtype;
  private pipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

  constructor(options: TransformersEmbedderOptions) {
    this.modelId = options.modelId;
    this.dim = options.dim;
    this.useWebGPU = options.useWebGPU ?? true;
    this.dtype = options.dtype ?? DEFAULT_DTYPE;
  }

  /** Lazily construct (and cache) the feature-extraction pipeline. */
  private async getPipeline(): Promise<FeatureExtractionPipeline> {
    if (this.pipelinePromise === null) {
      this.pipelinePromise = this.createPipeline();
    }
    return this.pipelinePromise;
  }

  private async createPipeline(): Promise<FeatureExtractionPipeline> {
    const transformers = await import("@huggingface/transformers");
    const fallback = nonGpuDevice();
    const device: Device = this.useWebGPU ? "webgpu" : fallback;
    try {
      const pipe = await transformers.pipeline("feature-extraction", this.modelId, {
        dtype: this.dtype,
        device,
      });
      return pipe as unknown as FeatureExtractionPipeline;
    } catch (error) {
      if (this.useWebGPU) {
        // WebGPU unavailable or regressed at runtime: fall back to the CPU/WASM
        // device silently with the same model, as the contingency rules require.
        const pipe = await transformers.pipeline("feature-extraction", this.modelId, {
          dtype: this.dtype,
          device: fallback,
        });
        return pipe as unknown as FeatureExtractionPipeline;
      }
      throw error;
    }
  }

  public async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return [];
    }
    const pipe = await this.getPipeline();
    const output = await pipe(texts, { pooling: "mean", normalize: true });
    return output.tolist().map((row) => Float32Array.from(row));
  }
}

/**
 * Deterministic feature-hashing embedder for offline tests. Each token is hashed
 * into one of `dim` buckets with a sign, term frequencies are accumulated, and
 * the vector is L2-normalized. Identical text always yields an identical vector,
 * and texts sharing vocabulary have positive cosine similarity.
 */
export class HashingEmbedder implements Embedder {
  public readonly dim: number;
  public readonly modelId = "hashing-embedder-v1";

  constructor(dim = 384) {
    this.dim = dim;
  }

  public async embed(texts: string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.embedOne(text));
  }

  private embedOne(text: string): Float32Array {
    const vector = new Float32Array(this.dim);
    const tokens = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
    for (const token of tokens) {
      const hash = parseInt(hashText(token), 16);
      const bucket = hash % this.dim;
      const sign = (hash & 0x100) === 0 ? 1 : -1;
      vector[bucket] += sign;
    }
    return l2Normalize(vector);
  }
}
