import { describe, expect, it } from "vitest";
import { Bm25Index, tokenizeForBm25 } from "../src/core/bm25";

describe("tokenizeForBm25", () => {
  it("lowercases and splits on non-alphanumerics", () => {
    expect(tokenizeForBm25("Reduce Coffee, please!")).toEqual(["reduce", "coffee", "please"]);
  });
});

describe("Bm25Index", () => {
  it("ranks documents with the exact query term first", () => {
    const index = new Bm25Index();
    index.add("a", "tapering off caffeine and coffee habits");
    index.add("b", "growing tomatoes in a summer garden");
    index.add("c", "a note about productivity and focus");

    const hits = index.search("caffeine", 3);
    expect(hits[0].id).toBe("a");
    expect(hits.find((h) => h.id === "b")).toBeUndefined();
  });

  it("rewards higher term frequency", () => {
    const index = new Bm25Index();
    index.add("dense", "coffee coffee coffee beans");
    index.add("sparse", "coffee and a long tail of other unrelated words here today");
    const hits = index.search("coffee", 2);
    expect(hits[0].id).toBe("dense");
  });

  it("supports incremental removal", () => {
    const index = new Bm25Index();
    index.add("a", "alpha beta");
    index.add("b", "beta gamma");
    expect(index.size).toBe(2);
    index.remove("a");
    expect(index.size).toBe(1);
    const hits = index.search("alpha", 5);
    expect(hits).toHaveLength(0);
  });

  it("returns no hits for out-of-vocabulary queries", () => {
    const index = new Bm25Index();
    index.add("a", "hello world");
    expect(index.search("nonexistentterm", 5)).toHaveLength(0);
  });
});
