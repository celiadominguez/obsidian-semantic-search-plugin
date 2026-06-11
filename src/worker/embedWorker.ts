/**
 * Embedding Web Worker entry point.
 *
 * Runs the transformers.js model off the UI thread so indexing never blocks the
 * editor. It loads the model once on `init`, then answers `embed` requests with
 * mean-pooled, L2-normalized vectors. This file is bundled into a string at
 * build time and instantiated from a Blob URL by the main thread.
 */

import { TransformersEmbedder } from "../core/embedder";
import type { WorkerRequest, WorkerResponse } from "./protocol";

let embedder: TransformersEmbedder | null = null;

const ctx = self as unknown as {
  postMessage(message: WorkerResponse): void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
};

function post(message: WorkerResponse): void {
  ctx.postMessage(message);
}

ctx.onmessage = async (event: MessageEvent<WorkerRequest>): Promise<void> => {
  const message = event.data;
  try {
    if (message.type === "init") {
      embedder = new TransformersEmbedder({
        modelId: message.modelId,
        dim: message.dim,
        useWebGPU: message.useWebGPU,
      });
      // Warm the pipeline with a trivial input so the first real batch is fast.
      await embedder.embed(["warm up"]);
      post({ type: "ready" });
      return;
    }

    if (message.type === "embed") {
      if (embedder === null) {
        post({ type: "error", requestId: message.requestId, message: "Embedder not initialized" });
        return;
      }
      const vectors = await embedder.embed(message.texts);
      post({
        type: "result",
        requestId: message.requestId,
        vectors: vectors.map((vector) => Array.from(vector)),
      });
    }
  } catch (error) {
    const requestId = message.type === "embed" ? message.requestId : null;
    post({
      type: "error",
      requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
