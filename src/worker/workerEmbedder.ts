/**
 * Main-thread bridge to the embedding Web Worker.
 *
 * Implements the `Embedder` interface by posting batches to the worker and
 * resolving per-request promises. If the worker cannot be created (e.g. a
 * platform without Blob workers), it transparently falls back to running the
 * transformers.js model on the main thread, so embedding always works even
 * though the worker is preferred for keeping the UI responsive.
 */

import workerCode from "inline:embed-worker";
import { TransformersEmbedder } from "../core/embedder";
import type { Embedder, EmbeddingModelId } from "../core/types";
import type { WorkerRequest, WorkerResponse } from "./protocol";

interface WorkerEmbedderOptions {
  modelId: EmbeddingModelId;
  dim: number;
  useWebGPU: boolean;
}

interface PendingRequest {
  resolve(vectors: Float32Array[]): void;
  reject(error: Error): void;
}

export class WorkerEmbedder implements Embedder {
  public readonly dim: number;
  public readonly modelId: string;

  private readonly options: WorkerEmbedderOptions;
  private worker: Worker | null = null;
  private fallback: TransformersEmbedder | null = null;
  private readyPromise: Promise<void> | null = null;
  private readonly pending = new Map<number, PendingRequest>();
  private nextRequestId = 1;

  constructor(options: WorkerEmbedderOptions) {
    this.options = options;
    this.modelId = options.modelId;
    this.dim = options.dim;
  }

  /** Embed texts, initializing the worker (or fallback) on first use. */
  public async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return [];
    }
    await this.ensureReady();
    if (this.worker === null) {
      // Fallback path: run on the main thread.
      return this.getFallback().embed(texts);
    }
    return this.dispatch(texts);
  }

  /** Tear down the worker and release the model. */
  public terminate(): void {
    this.worker?.terminate();
    this.worker = null;
    this.readyPromise = null;
    for (const request of this.pending.values()) {
      request.reject(new Error("Embedder terminated"));
    }
    this.pending.clear();
  }

  private getFallback(): TransformersEmbedder {
    if (this.fallback === null) {
      this.fallback = new TransformersEmbedder(this.options);
    }
    return this.fallback;
  }

  private ensureReady(): Promise<void> {
    if (this.readyPromise === null) {
      this.readyPromise = this.start();
    }
    return this.readyPromise;
  }

  private start(): Promise<void> {
    return new Promise<void>((resolve) => {
      let worker: Worker;
      try {
        const blob = new Blob([workerCode], { type: "application/javascript" });
        worker = new Worker(URL.createObjectURL(blob));
      } catch {
        // No worker available — resolve and let embed() use the fallback.
        this.worker = null;
        resolve();
        return;
      }

      worker.onmessage = (event: MessageEvent<WorkerResponse>): void => {
        const message = event.data;
        if (message.type === "ready") {
          this.worker = worker;
          resolve();
          return;
        }
        if (message.type === "result") {
          const request = this.pending.get(message.requestId);
          this.pending.delete(message.requestId);
          request?.resolve(message.vectors.map((vector) => Float32Array.from(vector)));
          return;
        }
        if (message.type === "error") {
          if (message.requestId === null) {
            // Initialization failure: drop to the main-thread fallback.
            worker.terminate();
            this.worker = null;
            resolve();
            return;
          }
          const request = this.pending.get(message.requestId);
          this.pending.delete(message.requestId);
          request?.reject(new Error(message.message));
        }
      };

      worker.onerror = (): void => {
        this.worker = null;
        resolve();
      };

      this.send(worker, { type: "init", ...this.options });
    });
  }

  private dispatch(texts: string[]): Promise<Float32Array[]> {
    const worker = this.worker;
    if (worker === null) {
      return this.getFallback().embed(texts);
    }
    return new Promise<Float32Array[]>((resolve, reject) => {
      const requestId = this.nextRequestId++;
      this.pending.set(requestId, { resolve, reject });
      this.send(worker, { type: "embed", requestId, texts });
    });
  }

  private send(worker: Worker, message: WorkerRequest): void {
    worker.postMessage(message);
  }
}
