/**
 * Acceptance criteria expressed as executable tests.
 *
 * These assert the shippable contract rather than internal details: a valid
 * manifest and consistent versions file, a buildable bundle, an end-to-end
 * indexing → ranking pipeline over the committed demo vault, an evaluation that
 * emits three-ranker metrics, persistence round-tripping, grounded Q&A with
 * resolvable citations, refusal on weak retrieval, and full settings coverage.
 *
 * To stay offline and fast they use the deterministic `HashingEmbedder`; the
 * real model numbers come from `npm run eval` and are recorded in the README.
 */

import { execSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { HashingEmbedder } from "../src/core/embedder";
import { defaultSettings } from "../src/core/config";
import { NoneGenerator } from "../src/core/qa";
import { ChatEngine } from "../src/core/chat";
import { VectorStore } from "../src/core/vectorStore";
import { rank } from "../src/core/hybridRanker";
import { buildIndex, evaluateAllModes, type Qrel } from "../eval/evaluate";
import type { NoteInput, VaultSeekSettings } from "../src/core/types";

const DEMO_VAULT_DIR = "demo-vault";
const SUBSET_SIZE = 60;
const DIM = 384;

function loadNote(fileName: string): NoteInput {
  const raw = readFileSync(join(DEMO_VAULT_DIR, fileName), "utf8");
  const titleMatch = /title:\s*"([^"]+)"/.exec(raw);
  return {
    path: fileName,
    title: titleMatch?.[1] ?? fileName.replace(/\.md$/, ""),
    content: raw,
    mtime: 0,
  };
}

function loadSubset(): NoteInput[] {
  return readdirSync(DEMO_VAULT_DIR)
    .filter((name) => name.endsWith(".md"))
    .sort()
    .slice(0, SUBSET_SIZE)
    .map(loadNote);
}

describe("MVP acceptance", () => {
  it("manifest.json is valid (id, semver, desktop-only, no numbering)", () => {
    const raw = readFileSync("manifest.json", "utf8");
    const manifest = JSON.parse(raw) as Record<string, unknown>;
    expect(manifest.id).toBe("vaultseek");
    expect(manifest.isDesktopOnly).toBe(true);
    expect(typeof manifest.version).toBe("string");
    expect(manifest.version as string).toMatch(/^\d+\.\d+\.\d+$/);
    expect(typeof manifest.minAppVersion).toBe("string");
    // No lab/phase numbering anywhere in the manifest.
    expect(raw).not.toMatch(/lab[-\s]?\d|phase[-\s]?\d/i);
  });

  it("`npm run build` produces a non-empty main.js", () => {
    execSync("npm run build", { stdio: "ignore" });
    expect(existsSync("main.js")).toBe(true);
    const stats = readFileSync("main.js");
    expect(stats.length).toBeGreaterThan(1000);
  }, 180000);

  it("indexes the demo vault and returns ranked hits for a known query", async () => {
    const notes = loadSubset();
    const embedder = new HashingEmbedder(DIM);
    const { chunkTokens, chunkOverlap } = defaultSettings();
    const { store, bm25 } = await buildIndex(embedder, notes, { chunkTokens, chunkOverlap });
    expect(store.size).toBeGreaterThan(0);

    // A known note's title is a query guaranteed to surface that note.
    const target = notes[0];
    const results = await rank({
      query: target.title,
      embedder,
      store,
      bm25,
      alpha: 0.6,
      mode: "hybrid",
      topK: 10,
    });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].chunk.notePath).toBe(target.path);
  });

  it("evaluation emits metrics for semantic, lexical, and hybrid", async () => {
    const notes = loadSubset();
    const embedder = new HashingEmbedder(DIM);
    const { chunkTokens, chunkOverlap } = defaultSettings();
    const { store, bm25 } = await buildIndex(embedder, notes, { chunkTokens, chunkOverlap });

    // Build a tiny qrel set from the subset: each note is "relevant" to its title.
    const qrels: Qrel[] = notes.slice(0, 10).map((note, i) => ({
      query_id: String(i),
      query: note.title,
      relevant: [note.path],
    }));
    const ranking = await evaluateAllModes(embedder, store, bm25, qrels, 0.6, 50);

    // Serialize and reparse to prove the metrics JSON is well-formed.
    const metrics = JSON.parse(JSON.stringify({ ranking }));
    for (const mode of ["semantic", "lexical", "hybrid"] as const) {
      expect(metrics.ranking[mode]).toBeDefined();
      expect(typeof metrics.ranking[mode].ndcg).toBe("number");
      expect(typeof metrics.ranking[mode].recall).toBe("number");
      expect(metrics.ranking[mode].ndcg).toBeGreaterThanOrEqual(0);
      expect(metrics.ranking[mode].recall).toBeLessThanOrEqual(1);
    }
  });
});

describe("Hardening acceptance", () => {
  it("persisted index reloads and reproduces results", async () => {
    const notes = loadSubset();
    const embedder = new HashingEmbedder(DIM);
    const { chunkTokens, chunkOverlap } = defaultSettings();
    const { store } = await buildIndex(embedder, notes, { chunkTokens, chunkOverlap });

    const [query] = await embedder.embed([notes[3].title]);
    const before = await store.search(query, 20);
    const { vectors, sidecar } = store.toBlob();
    const reloaded = VectorStore.fromBlob(vectors, sidecar, 100000);
    const after = await reloaded.search(query, 20);
    expect(after).toEqual(before);
  });

  it("README records the eval Results with the three rankers and an honest verdict", () => {
    const readme = readFileSync("README.md", "utf8");
    expect(readme).toMatch(/##\s*Results/i);
    expect(readme.toLowerCase()).toContain("semantic");
    expect(readme.toLowerCase()).toContain("lexical");
    expect(readme.toLowerCase()).toContain("hybrid");
    expect(readme.toLowerCase()).toMatch(/ndcg/);
    expect(readme.toLowerCase()).toMatch(/verdict/);
  });

  it("chat citations resolve to real demo-vault note paths", async () => {
    const notes = loadSubset();
    const embedder = new HashingEmbedder(DIM);
    const { chunkTokens, chunkOverlap } = defaultSettings();
    const { store, bm25 } = await buildIndex(embedder, notes, { chunkTokens, chunkOverlap });

    const engine = new ChatEngine({
      embedder,
      store,
      bm25,
      generator: new NoneGenerator(),
      alpha: 0.6,
      similarityFloor: 0.1,
    });
    const { message } = await engine.ask(notes[0].title);
    expect(message.refused).toBe(false);
    expect(message.citations.length).toBeGreaterThan(0);
    for (const citation of message.citations) {
      expect(existsSync(join(DEMO_VAULT_DIR, citation))).toBe(true);
    }
  });

  it("offline chat refuses when retrieval is below the confidence floor", async () => {
    const notes = loadSubset();
    const embedder = new HashingEmbedder(DIM);
    const { chunkTokens, chunkOverlap } = defaultSettings();
    const { store, bm25 } = await buildIndex(embedder, notes, { chunkTokens, chunkOverlap });

    const engine = new ChatEngine({
      embedder,
      store,
      bm25,
      generator: new NoneGenerator(),
      alpha: 0.6,
    });
    const { message } = await engine.ask("qwxz vbnmqp plokju ytrewq zxcvbn asdfgh");
    expect(message.refused).toBe(true);
    expect(message.citations).toHaveLength(0);
  });

  it("settings tab covers every configuration key", () => {
    const source = readFileSync(join("src", "obsidian", "SettingsTab.ts"), "utf8");
    const settings: VaultSeekSettings = defaultSettings();
    for (const key of Object.keys(settings)) {
      expect(source, `SettingsTab is missing settings.${key}`).toContain(`settings.${key}`);
    }
  });

  it("versions.json is consistent with manifest.json", () => {
    const manifest = JSON.parse(readFileSync("manifest.json", "utf8")) as {
      version: string;
      minAppVersion: string;
    };
    const versions = JSON.parse(readFileSync("versions.json", "utf8")) as Record<string, string>;
    expect(versions[manifest.version]).toBe(manifest.minAppVersion);
  });
});
