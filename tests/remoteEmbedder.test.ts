import { describe, expect, it, vi } from "vitest";
import { RemoteEmbedder, detectRemoteEmbeddingDim } from "../src/core/remoteEmbedder";
import type { HttpClient } from "../src/core/http";

/** An HttpClient stub that records calls and returns a canned JSON body. */
function stubHttp(body: unknown, ok = true, status = 200) {
  const calls: Array<{ url: string; body: unknown }> = [];
  const http: HttpClient = async (url, init) => {
    calls.push({ url, body: init?.body === undefined ? undefined : JSON.parse(init.body) });
    return { ok, status, json: async () => body };
  };
  return { http, calls };
}

/** Length of a vector, used to assert normalization. */
function norm(vector: Float32Array): number {
  return Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
}

describe("RemoteEmbedder (OpenAI-compatible)", () => {
  const options = {
    protocol: "openai" as const,
    endpoint: "http://localhost:1234/v1",
    model: "text-embed",
    dim: 3,
  };

  it("posts to /embeddings and returns L2-normalized vectors", async () => {
    const { http, calls } = stubHttp({
      data: [
        { index: 0, embedding: [3, 0, 0] },
        { index: 1, embedding: [0, 4, 0] },
      ],
    });
    const vectors = await new RemoteEmbedder({ ...options, http }).embed(["a", "b"]);

    expect(calls[0].url).toBe("http://localhost:1234/v1/embeddings");
    expect(calls[0].body).toEqual({ model: "text-embed", input: ["a", "b"] });
    expect(vectors).toHaveLength(2);
    expect(norm(vectors[0])).toBeCloseTo(1);
    expect(Array.from(vectors[0])).toEqual([1, 0, 0]);
  });

  it("reorders rows by the index field", async () => {
    const { http } = stubHttp({
      data: [
        { index: 1, embedding: [0, 1, 0] },
        { index: 0, embedding: [1, 0, 0] },
      ],
    });
    const vectors = await new RemoteEmbedder({ ...options, http }).embed(["first", "second"]);
    expect(Array.from(vectors[0])).toEqual([1, 0, 0]);
    expect(Array.from(vectors[1])).toEqual([0, 1, 0]);
  });

  it("sends a bearer token only when a key is configured", async () => {
    const seen: Array<Record<string, string> | undefined> = [];
    const http: HttpClient = async (_url, init) => {
      seen.push(init?.headers);
      return { ok: true, status: 200, json: async () => ({ data: [{ embedding: [1, 0, 0] }] }) };
    };
    await new RemoteEmbedder({ ...options, http }).embed(["x"]);
    await new RemoteEmbedder({ ...options, http, apiKey: "secret" }).embed(["x"]);

    expect(seen[0]?.Authorization).toBeUndefined();
    expect(seen[1]?.Authorization).toBe("Bearer secret");
  });

  it("throws when the server's dimension does not match the configured one", async () => {
    const { http } = stubHttp({ data: [{ embedding: [1, 2] }] });
    await expect(new RemoteEmbedder({ ...options, http }).embed(["x"])).rejects.toThrow(
      /2-dimensional vectors but the configured dimension is 3/,
    );
  });

  it("throws when the server returns the wrong number of vectors", async () => {
    const { http } = stubHttp({ data: [{ embedding: [1, 0, 0] }] });
    await expect(new RemoteEmbedder({ ...options, http }).embed(["a", "b"])).rejects.toThrow(
      /returned 1 vectors for 2 inputs/,
    );
  });

  it("surfaces a non-2xx response", async () => {
    const { http } = stubHttp({}, false, 503);
    await expect(new RemoteEmbedder({ ...options, http }).embed(["x"])).rejects.toThrow(
      /Embeddings request failed: 503/,
    );
  });

  it("skips the request entirely for an empty batch", async () => {
    const spy = vi.fn();
    const http: HttpClient = async (...args) => {
      spy(...args);
      return { ok: true, status: 200, json: async () => ({}) };
    };
    await expect(new RemoteEmbedder({ ...options, http }).embed([])).resolves.toEqual([]);
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("RemoteEmbedder (Ollama)", () => {
  const options = {
    protocol: "ollama" as const,
    endpoint: "http://localhost:11434",
    model: "nomic-embed-text",
    dim: 3,
  };

  it("posts to /api/embed and reads the embeddings array", async () => {
    const { http, calls } = stubHttp({ embeddings: [[0, 0, 5]] });
    const vectors = await new RemoteEmbedder({ ...options, http }).embed(["a"]);

    expect(calls[0].url).toBe("http://localhost:11434/api/embed");
    expect(calls[0].body).toEqual({ model: "nomic-embed-text", input: ["a"] });
    expect(Array.from(vectors[0])).toEqual([0, 0, 1]);
  });

  it("trims a trailing slash from the endpoint", async () => {
    const { http, calls } = stubHttp({ embeddings: [[1, 0, 0]] });
    await new RemoteEmbedder({ ...options, endpoint: "http://localhost:11434/", http }).embed([
      "a",
    ]);
    expect(calls[0].url).toBe("http://localhost:11434/api/embed");
  });
});

describe("modelId", () => {
  it("encodes protocol, endpoint and model so a switch invalidates the index", () => {
    const { http } = stubHttp({});
    const base = { endpoint: "http://h/v1", model: "m", dim: 3, http };
    const a = new RemoteEmbedder({ ...base, protocol: "openai" }).modelId;
    const b = new RemoteEmbedder({ ...base, protocol: "ollama" }).modelId;
    const c = new RemoteEmbedder({ ...base, protocol: "openai", model: "other" }).modelId;

    expect(a).not.toBe(b);
    expect(a).not.toBe(c);
    expect(a).toContain("http://h/v1");
  });
});

describe("detectRemoteEmbeddingDim", () => {
  it("reports the length the server actually returns", async () => {
    const { http } = stubHttp({ data: [{ embedding: [1, 2, 3, 4, 5] }] });
    const dim = await detectRemoteEmbeddingDim({
      protocol: "openai",
      endpoint: "http://localhost:1234/v1",
      model: "m",
      http,
    });
    expect(dim).toBe(5);
  });

  it("throws when the probe comes back empty", async () => {
    const { http } = stubHttp({ data: [] });
    await expect(
      detectRemoteEmbeddingDim({
        protocol: "openai",
        endpoint: "http://localhost:1234/v1",
        model: "m",
        http,
      }),
    ).rejects.toThrow(/no vector for the probe/);
  });
});
