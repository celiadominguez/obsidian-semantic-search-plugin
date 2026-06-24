import { describe, expect, it } from "vitest";
import { blend, normalizeScores, rank } from "../src/core/hybridRanker";
import { HashingEmbedder, l2Normalize } from "../src/core/embedder";
import { Bm25Index } from "../src/core/bm25";
import { VectorStore } from "../src/core/vectorStore";
import { hashText } from "../src/core/hash";
import type { Chunk, Embedder } from "../src/core/types";

describe("normalizeScores", () => {
  it("min-max normalizes into [0, 1]", () => {
    const norm = normalizeScores(
      new Map([
        ["a", 2],
        ["b", 4],
        ["c", 6],
      ]),
    );
    expect(norm.get("a")).toBe(0);
    expect(norm.get("b")).toBe(0.5);
    expect(norm.get("c")).toBe(1);
  });

  it("maps a flat distribution to 1", () => {
    const norm = normalizeScores(
      new Map([
        ["a", 3],
        ["b", 3],
      ]),
    );
    expect(norm.get("a")).toBe(1);
    expect(norm.get("b")).toBe(1);
  });
});

describe("blend", () => {
  it("applies score = alpha*cosine + (1-alpha)*bm25_norm", () => {
    const semantic = new Map([
      ["a", 0.8],
      ["b", 0.2],
    ]);
    const lexical = new Map([
      ["a", 0.0],
      ["b", 1.0],
    ]);
    const blended = blend(semantic, lexical, 0.6);
    const byId = new Map(blended.map((b) => [b.id, b]));

    // a: 0.6*0.8 + 0.4*0.0 = 0.48 ; b: 0.6*0.2 + 0.4*1.0 = 0.52
    expect(byId.get("a")!.score).toBeCloseTo(0.48, 6);
    expect(byId.get("b")!.score).toBeCloseTo(0.52, 6);
    // Sorted descending, so b (0.52) outranks a (0.48).
    expect(blended[0].id).toBe("b");
  });

  it("treats missing signals as zero", () => {
    const blended = blend(new Map([["only", 0.5]]), new Map(), 0.6);
    expect(blended[0].semanticScore).toBe(0.5);
    expect(blended[0].lexicalScore).toBe(0);
    expect(blended[0].score).toBeCloseTo(0.3, 6);
  });
});

describe("rank integration", () => {
  function chunk(id: string, notePath: string, text: string): Chunk {
    return { id, notePath, noteTitle: notePath, heading: "", ordinal: 0, text };
  }

  async function makeEngine() {
    const embedder = new HashingEmbedder(64);
    const store = new VectorStore({ dim: 64, modelId: embedder.modelId, hnswThreshold: 1000 });
    const bm25 = new Bm25Index();
    const docs: Chunk[] = [
      chunk("c0", "caffeine.md", "tapering off caffeine and reducing coffee consumption"),
      chunk("c1", "garden.md", "growing tomatoes and peppers in a summer vegetable garden"),
      chunk("c2", "focus.md", "techniques for deep focus and sustained concentration"),
    ];
    const vectors = await embedder.embed(docs.map((d) => d.text));
    docs.forEach((d, i) => {
      store.upsert({ chunk: d, hash: hashText(d.text), vector: vectors[i] });
      bm25.add(d.id, d.text);
    });
    return { embedder, store, bm25 };
  }

  it("ranks the on-topic note first in every mode", async () => {
    const { embedder, store, bm25 } = await makeEngine();
    for (const mode of ["semantic", "lexical", "hybrid"] as const) {
      const results = await rank({
        query: "how do I reduce coffee",
        embedder,
        store,
        bm25,
        alpha: 0.6,
        mode,
        topK: 3,
      });
      expect(results[0].chunk.notePath, `mode=${mode}`).toBe("caffeine.md");
      expect(results[0].snippet.length).toBeGreaterThan(0);
    }
  });

  it("prepends the query instruction to the embedded query but not to BM25", async () => {
    // Records exactly what text the embedder receives.
    class RecordingEmbedder implements Embedder {
      public readonly dim = 8;
      public readonly modelId = "recording";
      public readonly seen: string[] = [];
      public async embed(texts: string[]): Promise<Float32Array[]> {
        this.seen.push(...texts);
        return texts.map(() => l2Normalize(new Float32Array(this.dim).fill(1)));
      }
    }
    const embedder = new RecordingEmbedder();
    const store = new VectorStore({ dim: 8, modelId: embedder.modelId, hnswThreshold: 1000 });
    const bm25 = new Bm25Index();
    const c = chunk("c0", "n.md", "alpha beta");
    const [vec] = await embedder.embed(["alpha beta"]);
    store.upsert({ chunk: c, hash: hashText(c.text), vector: vec });
    bm25.add(c.id, c.text);
    embedder.seen.length = 0;

    await rank({
      query: "alpha",
      embedder,
      store,
      bm25,
      alpha: 0.6,
      mode: "hybrid",
      topK: 1,
      queryInstruction: "PREFIX: ",
    });
    // The embedded query carries the instruction…
    expect(embedder.seen).toContain("PREFIX: alpha");
    // …and BM25 still matches the raw term (no leakage into lexical scoring).
    expect(bm25.search("alpha", 1)).toHaveLength(1);
  });

  it("populates both signal scores in hybrid mode", async () => {
    const { embedder, store, bm25 } = await makeEngine();
    const [top] = await rank({
      query: "summer garden tomatoes",
      embedder,
      store,
      bm25,
      alpha: 0.6,
      mode: "hybrid",
      topK: 1,
    });
    expect(top.chunk.notePath).toBe("garden.md");
    expect(top.lexicalScore).toBeGreaterThan(0);
    expect(top.semanticScore).toBeGreaterThan(0);
  });
});
