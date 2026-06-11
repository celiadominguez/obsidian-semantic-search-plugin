/**
 * Message protocol shared between the main thread and the embedding Web Worker.
 * Kept in its own module so both sides import identical, type-checked shapes.
 */

import type { EmbeddingModelId } from "../core/types";

/** Main thread → worker: one-time model initialization. */
export interface InitMessage {
  type: "init";
  modelId: EmbeddingModelId;
  dim: number;
  useWebGPU: boolean;
}

/** Main thread → worker: embed a batch of texts, tagged with a request id. */
export interface EmbedMessage {
  type: "embed";
  requestId: number;
  texts: string[];
}

export type WorkerRequest = InitMessage | EmbedMessage;

/** Worker → main thread: model is loaded and ready to embed. */
export interface ReadyMessage {
  type: "ready";
}

/** Worker → main thread: embedding result for a request, as plain number arrays. */
export interface ResultMessage {
  type: "result";
  requestId: number;
  vectors: number[][];
}

/** Worker → main thread: an error, optionally tied to a specific request. */
export interface ErrorMessage {
  type: "error";
  requestId: number | null;
  message: string;
}

export type WorkerResponse = ReadyMessage | ResultMessage | ErrorMessage;
