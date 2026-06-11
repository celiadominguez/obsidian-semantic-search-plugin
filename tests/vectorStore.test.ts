import { describe, expect, it } from "vitest";
import { HashingEmbedder } from "../src/core/embedder";
import { VectorStore } from "../src/core/vectorStore";
import type { Chunk, VectorRecord } from "../src/core/types";

const DIM = 16;

function chunk(id: string, notePath: string, text: string): Chunk {
  return { id, notePath, noteTitle: notePath, heading: "", ordinal: 0, text };
}

async function buildRecords(): Promise<VectorRecord[]> {
  const embedder = new HashingEmbedder(DIM);
  const texts = [
    "tapering off caffeine and coffee",
    "growing tomatoes in the garden",
    "a guide to deep focus and productivity",
    "machine learning embeddings and vectors",
    "morning routines and waking up early",
    "reducing coffee intake gradually over weeks",
  ];
  const vectors = await embedder.embed(texts);
  return texts.map((text, i) => ({
    chunk: chunk(`c${i}`, `note${i}.md`, text),
    hash: `h${i}`,
    vector: vectors[i],
  }));
}

describe("VectorStore exact search", () => {
  it("returns the most similar chunk first", async () => {
    const store = new VectorStore({ dim: DIM, modelId: "test", hnswThreshold: 100 });
    const records = await buildRecords();
    records.forEach((r) => store.upsert(r));

    // Querying with a stored vector should return that chunk at rank 1.
    const hits = await store.search(records[0].vector, 3);
    expect(hits[0].id).toBe("c0");
    expect(hits[0].score).toBeCloseTo(1, 5);
    expect(hits).toHaveLength(3);
  });

  it("removes a note's chunks and updates membership", async () => {
    const store = new VectorStore({ dim: DIM, modelId: "test", hnswThreshold: 100 });
    const records = await buildRecords();
    records.forEach((r) => store.upsert(r));
    expect(store.size).toBe(records.length);

    store.removeNote("note0.md");
    expect(store.has("c0")).toBe(false);
    expect(store.size).toBe(records.length - 1);
  });

  it("rejects vectors of the wrong dimension", async () => {
    const store = new VectorStore({ dim: DIM, modelId: "test", hnswThreshold: 100 });
    expect(() =>
      store.upsert({ chunk: chunk("x", "x.md", "x"), hash: "h", vector: new Float32Array(4) }),
    ).toThrow(/dimension/);
  });
});

describe("VectorStore persistence", () => {
  it("round-trips vectors and reproduces search results exactly", async () => {
    const store = new VectorStore({ dim: DIM, modelId: "test", hnswThreshold: 100 });
    const records = await buildRecords();
    records.forEach((r) => store.upsert(r));

    const query = records[2].vector;
    const before = await store.search(query, records.length);

    const { vectors, sidecar } = store.toBlob();
    const reloaded = VectorStore.fromBlob(vectors, sidecar, 100);
    const after = await reloaded.search(query, records.length);

    expect(reloaded.size).toBe(store.size);
    // Identical scores (not just ids) prove the vectors survived the round-trip
    // exactly, since cosine is sensitive to any change in the stored vectors.
    expect(after).toEqual(before);
    expect(after[0].score).toBeCloseTo(1, 5);
  });

  it("rejects a blob whose length disagrees with the sidecar", async () => {
    const store = new VectorStore({ dim: DIM, modelId: "test", hnswThreshold: 100 });
    const records = await buildRecords();
    records.forEach((r) => store.upsert(r));
    const { sidecar } = store.toBlob();
    expect(() => VectorStore.fromBlob(new ArrayBuffer(8), sidecar, 100)).toThrow(/blob length/);
  });
});

describe("VectorStore HNSW parity", () => {
  it("returns the same top-k as exact cosine on a small set", async () => {
    const records = await buildRecords();
    const exact = new VectorStore({ dim: DIM, modelId: "test", hnswThreshold: 1000 });
    const approx = new VectorStore({ dim: DIM, modelId: "test", hnswThreshold: 1 });
    records.forEach((r) => {
      exact.upsert(r);
      approx.upsert(r);
    });
    expect(approx.usesHnsw).toBe(true);

    const query = records[5].vector;
    const exactHits = (await exact.search(query, 3)).map((h) => h.id);
    // Above threshold this uses HNSW when the WASM module loads, and otherwise
    // falls back to exact cosine; either way the small-set top-k must match.
    const approxHits = (await approx.search(query, 3)).map((h) => h.id);
    expect(approxHits).toEqual(exactHits);
  });
});
