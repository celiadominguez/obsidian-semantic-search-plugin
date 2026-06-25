/**
 * Embedding utilities and implementations.
 *
 * Two concerns live here: pure vector math (L2-normalization, cosine) shared by
 * every code path, and two `Embedder` implementations:
 *
 *  - `TransformersEmbedder` — the real on-device model via transformers.js. It
 *    runs unchanged in Node (for the offline eval) and in Obsidian's renderer
 *    (for the plugin), on the CPU (WASM) execution backend.
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
let transformersPromise: Promise<typeof import("@huggingface/transformers")> | null = null;

async function importTransformersOnce(): Promise<typeof import("@huggingface/transformers")> {
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
 * Import transformers.js exactly once, serialized through a module-level promise.
 *
 * The {@link importTransformersOnce} body temporarily mutates the global
 * `process.release`, so two concurrent embedder initializations could otherwise
 * interleave — one restoring `release` while another is still mid-import.
 * Caching the in-flight promise guarantees a single spoof window for the whole
 * process and lets every caller share the one resolved module.
 */
async function importTransformers(): Promise<typeof import("@huggingface/transformers")> {
  if (transformersPromise === null) {
    transformersPromise = importTransformersOnce();
  }
  return transformersPromise;
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
function configureOnnxRuntime(transformers: unknown, wasmBinary?: ArrayBuffer): void {
  const env = (transformers as { env?: unknown }).env;
  const wasm = (
    env as
      | {
          backends?: {
            onnx?: { wasm?: { numThreads?: number; proxy?: boolean; wasmBinary?: ArrayBuffer } };
          };
        }
      | undefined
  )?.backends?.onnx?.wasm;
  if (wasm !== undefined) {
    wasm.numThreads = 1;
    wasm.proxy = false;
    // Hand ORT the runtime bytes directly so it never fetches the .wasm from a
    // CDN (Obsidian prohibits downloading remote code). Undefined under real Node
    // (the eval), which uses onnxruntime-node and ignores these WASM settings.
    if (wasmBinary !== undefined) {
      wasm.wasmBinary = wasmBinary;
    }
  }
}

/** Reads a model file (path relative to the local model folder), or undefined. */
export type LocalModelReader = (relativePath: string) => Promise<ArrayBuffer | undefined>;

/**
 * Extract a model-relative file path (e.g. `Xenova/bge-small-en-v1.5/onnx/
 * model_quantized.onnx`) from a transformers.js model-file request URL, so it can
 * be read from the user's local model folder. Returns undefined when the URL is
 * not for this model. Hugging Face URLs carry a `resolve/<revision>/` infix that
 * a plain on-disk layout omits, so it is stripped.
 */
export function modelFileRelativePath(url: string, modelId: string): string | undefined {
  const marker = `${modelId}/`;
  const at = url.indexOf(marker);
  if (at === -1) {
    return undefined;
  }
  const rest = url.slice(at + marker.length).replace(/^resolve\/[^/]+\//, "");
  return `${modelId}/${rest}`;
}

/**
 * Point transformers.js at locally-stored model files instead of the Hugging Face
 * CDN. We register a custom cache whose `match` serves each requested model file
 * from `readFile` (an injected, Obsidian-backed reader) and disable remote model
 * fetching, so embedding runs fully offline. Opt-in: only wired up when the user
 * configures a local model folder. A miss surfaces as a load error (rather than a
 * silent download) because remote models are disabled.
 */
function configureLocalModel(
  transformers: unknown,
  readFile: LocalModelReader,
  modelId: string,
): void {
  const env = (transformers as { env?: Record<string, unknown> }).env;
  if (env === undefined) {
    return;
  }
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.useCustomCache = true;
  env.customCache = {
    async match(request: string | { url?: string }): Promise<Response | undefined> {
      const url = typeof request === "string" ? request : (request.url ?? "");
      const relativePath = modelFileRelativePath(url, modelId);
      if (relativePath === undefined) {
        return undefined;
      }
      const bytes = await readFile(relativePath);
      return bytes === undefined ? undefined : new Response(bytes);
    },
    async put(): Promise<void> {
      // Offline: there is nothing to write back.
    },
  };
}

/** Default quantization: int8, keeping the bge-small model around 33 MB. */
const DEFAULT_DTYPE: Dtype = "q8";

interface TransformersEmbedderOptions {
  modelId: EmbeddingModelId;
  dim: number;
  /** Exact Hugging Face commit to load, pinning weights for reproducibility. */
  revision: string;
  dtype?: Dtype;
  /**
   * Supplies the ONNX Runtime WASM bytes so the runtime loads locally instead of
   * fetching from a CDN. Read lazily (only when the pipeline is first built).
   * Omitted under real Node (the eval), which uses the native onnxruntime-node.
   */
  getWasmBinary?: () => Promise<ArrayBuffer | undefined>;
  /**
   * When set, loads the model from local files via this reader (and disables
   * remote model fetching) instead of downloading from Hugging Face. Opt-in,
   * fully-offline path. Omitted for the default download-and-cache behaviour.
   */
  readLocalModelFile?: LocalModelReader;
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
  private readonly revision: string;
  private readonly dtype: Dtype;
  private readonly getWasmBinary?: () => Promise<ArrayBuffer | undefined>;
  private readonly readLocalModelFile?: LocalModelReader;
  private pipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

  constructor(options: TransformersEmbedderOptions) {
    this.modelId = options.modelId;
    this.revision = options.revision;
    this.dim = options.dim;
    this.dtype = options.dtype ?? DEFAULT_DTYPE;
    this.getWasmBinary = options.getWasmBinary;
    this.readLocalModelFile = options.readLocalModelFile;
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
    const wasmBinary = this.getWasmBinary ? await this.getWasmBinary() : undefined;
    configureOnnxRuntime(transformers, wasmBinary);
    if (this.readLocalModelFile !== undefined) {
      configureLocalModel(transformers, this.readLocalModelFile, this.modelId);
    }
    const pipe = await transformers.pipeline("feature-extraction", this.modelId, {
      dtype: this.dtype,
      device: nonGpuDevice(),
      revision: this.revision,
    });
    return pipe as unknown as FeatureExtractionPipeline;
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
