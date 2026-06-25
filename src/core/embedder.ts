/**
 * Embedding utilities and implementations.
 *
 * Two concerns live here: pure vector math (L2-normalization, cosine) shared by
 * every code path, and two `Embedder` implementations:
 *
 *  - `TransformersEmbedder` — the real on-device model via transformers.js. It
 *    runs unchanged in Node (for the offline eval) and in Obsidian's renderer
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
 * True only under a *real* Node process (the eval), not Obsidian's Electron
 * renderer. Checking `process.versions.node` is not enough: the renderer exposes
 * Node's `process` too, which is exactly what misleads transformers.js. The
 * renderer additionally has a DOM `window`; a real Node process does not.
 */
function isRealNode(): boolean {
  return (
    typeof window === "undefined" &&
    typeof process !== "undefined" &&
    typeof process.versions?.node === "string"
  );
}

/**
 * Non-GPU device for the current runtime: transformers.js exposes "wasm" in the
 * browser but "cpu" under Node, so the fallback device differs by environment.
 * This keeps the same model working in the plugin and in the eval.
 */
function nonGpuDevice(): Device {
  return isRealNode() ? "cpu" : "wasm";
}

/**
 * Import transformers.js so it selects its **web** backend inside an Electron
 * renderer.
 *
 * Obsidian's renderer exposes Node's `process`, so transformers.js's
 * `IS_NODE_ENV` check (`process.release.name === 'node'`) is true and it reaches
 * for `onnxruntime-node` (absent in the browser bundle), leaving
 * `InferenceSession` undefined. We therefore present a non-"node" `process.release`
 * for the duration of the module's one-time initialization, which makes
 * transformers take its web path — wiring up onnxruntime-web *and* the supported
 * device list (`wasm`/`webgpu`). The original `release` is restored immediately
 * after. No-op under real Node (the eval), where onnxruntime-node is correct.
 */
async function importTransformers(): Promise<typeof import("@huggingface/transformers")> {
  const proc = typeof process !== "undefined" ? process : undefined;
  const release = proc?.release;
  const spoof =
    !isRealNode() && release !== undefined && release !== null && release.name === "node";
  if (spoof && proc !== undefined) {
    Object.defineProperty(proc, "release", {
      value: { ...release, name: "electron" },
      configurable: true,
    });
  }
  try {
    return await import("@huggingface/transformers");
  } finally {
    if (spoof && proc !== undefined) {
      Object.defineProperty(proc, "release", { value: release, configurable: true });
    }
  }
}

/**
 * Configure the ONNX Runtime WASM backend for restrictive renderers.
 *
 * Obsidian's Electron renderer is not cross-origin isolated, so `SharedArrayBuffer`
 * is unavailable and ORT's default multi-threaded WASM backend fails to register —
 * which surfaces downstream as "Cannot read properties of undefined (reading
 * 'create')" when transformers.js calls `InferenceSession.create`. Forcing a
 * single thread and disabling the proxy worker uses the backend that works there.
 * In Node this targets onnxruntime-node, where these WASM settings are simply
 * ignored, so it is safe on every code path.
 */
function configureOnnxRuntime(transformers: unknown): void {
  const env = (transformers as { env?: unknown }).env;
  const wasm = (
    env as { backends?: { onnx?: { wasm?: { numThreads?: number; proxy?: boolean } } } } | undefined
  )?.backends?.onnx?.wasm;
  if (wasm !== undefined) {
    wasm.numThreads = 1;
    wasm.proxy = false;
  }
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
    const transformers = await importTransformers();
    configureOnnxRuntime(transformers);
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
