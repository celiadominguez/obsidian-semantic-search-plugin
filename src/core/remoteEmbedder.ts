/**
 * Embedder backed by an external embeddings server.
 *
 * An opt-in alternative to the bundled on-device model: instead of running
 * transformers.js in the renderer, each batch of chunks is POSTed to a server
 * that returns vectors. Two wire formats are supported:
 *
 *  - `openai`: `POST {endpoint}/embeddings` with `{model, input[]}`, returning
 *    `{data: [{embedding, index}]}`. Covers LM Studio, Text Embeddings Inference,
 *    Infinity, and hosted OpenAI-compatible APIs.
 *  - `ollama`: `POST {endpoint}/api/embed` with `{model, input[]}`, returning
 *    `{embeddings: number[][]}`.
 *
 * Privacy: unlike the on-device path, this sends note text to the configured
 * endpoint. Pointed at localhost (Ollama / LM Studio) nothing leaves the machine;
 * pointed at a hosted API it does, which the settings UI states explicitly.
 *
 * Like every `Embedder`, the returned vectors are L2-normalized so cosine
 * similarity is a plain dot product downstream.
 */

import { l2Normalize } from "./embedder";
import type { HttpClient } from "./http";
import type { Embedder, RemoteEmbeddingProtocol } from "./types";

/** Drop a trailing slash so endpoints concatenate cleanly. */
function trimSlash(url: string): string {
  return url.replace(/\/$/, "");
}

export interface RemoteEmbedderOptions {
  protocol: RemoteEmbeddingProtocol;
  /** Base URL of the server, e.g. `http://localhost:1234/v1`. */
  endpoint: string;
  /** Model name the server should embed with. */
  model: string;
  /** Expected output dimensionality; a mismatch is reported, never ignored. */
  dim: number;
  /** Optional bearer token, sent only to this endpoint. */
  apiKey?: string;
  http: HttpClient;
}

export class RemoteEmbedder implements Embedder {
  public readonly dim: number;
  /**
   * Recorded in the index sidecar so switching server, protocol, or model
   * invalidates the persisted vectors and forces a clean re-embed.
   */
  public readonly modelId: string;
  private readonly protocol: RemoteEmbeddingProtocol;
  private readonly endpoint: string;
  private readonly model: string;
  private readonly apiKey: string;
  private readonly http: HttpClient;

  constructor(options: RemoteEmbedderOptions) {
    this.protocol = options.protocol;
    this.endpoint = trimSlash(options.endpoint);
    this.model = options.model;
    this.dim = options.dim;
    this.apiKey = options.apiKey ?? "";
    this.http = options.http;
    this.modelId = `remote:${options.protocol}:${this.endpoint}:${options.model}`;
  }

  /** Fetch raw vectors from the server, without the dimension assertion. */
  private async embedRaw(texts: string[]): Promise<number[][]> {
    return this.protocol === "ollama" ? this.embedOllama(texts) : this.embedOpenAi(texts);
  }

  /**
   * Embed a probe string to discover this model's dimensionality, so the user
   * does not have to look it up. Bypasses the configured-dimension check.
   */
  public async probeDim(): Promise<number> {
    const [vector] = await this.embedRaw(["test"]);
    if (vector === undefined || vector.length === 0) {
      throw new Error("Embeddings server returned no vector for the probe request");
    }
    return vector.length;
  }

  public async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return [];
    }
    const vectors = await this.embedRaw(texts);

    if (vectors.length !== texts.length) {
      throw new Error(
        `Embeddings server returned ${vectors.length} vectors for ${texts.length} inputs`,
      );
    }
    return vectors.map((row) => {
      if (row.length !== this.dim) {
        throw new Error(
          `Embeddings server returned ${row.length}-dimensional vectors but the ` +
            `configured dimension is ${this.dim}. Update the dimension in settings ` +
            `(and re-index) to match the model.`,
        );
      }
      return l2Normalize(Float32Array.from(row));
    });
  }

  /** Headers for the request, adding bearer auth only when a key is configured. */
  private headers(): Record<string, string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey.length > 0) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async post(url: string, body: unknown): Promise<unknown> {
    const response = await this.http(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`Embeddings request failed: ${response.status}`);
    }
    return response.json();
  }

  private async embedOpenAi(texts: string[]): Promise<number[][]> {
    const data = (await this.post(`${this.endpoint}/embeddings`, {
      model: this.model,
      input: texts,
    })) as { data?: Array<{ embedding?: number[]; index?: number }> };
    const rows = data.data ?? [];
    // `index` is authoritative when present: the spec allows any ordering.
    const ordered = [...rows].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    return ordered.map((row) => row.embedding ?? []);
  }

  private async embedOllama(texts: string[]): Promise<number[][]> {
    const data = (await this.post(`${this.endpoint}/api/embed`, {
      model: this.model,
      input: texts,
    })) as { embeddings?: number[][] };
    return data.embeddings ?? [];
  }
}

/**
 * Discover a remote model's dimensionality by embedding a probe string, so the
 * user does not have to look it up. Returns the vector length.
 */
export async function detectRemoteEmbeddingDim(
  options: Omit<RemoteEmbedderOptions, "dim">,
): Promise<number> {
  // The probe bypasses the dimension check, so the placeholder here is unused.
  return new RemoteEmbedder({ ...options, dim: 0 }).probeDim();
}
