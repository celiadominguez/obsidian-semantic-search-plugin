/**
 * Offline retrieval-quality evaluation CLI.
 *
 * Indexes the committed demo vault headlessly with the real on-device embedding
 * model and reports nDCG@10 and recall@10 for semantic, lexical, and hybrid
 * ranking against the SciFact qrels, plus an answer-grounding sanity check on the
 * WikiQA slice. Results are written to `eval/results/metrics_<timestamp>.json`.
 *
 * Everything runs locally: the only network access is transformers.js fetching
 * the embedding model once (then cached on disk), exactly as in the plugin. The
 * reusable scoring logic lives in `evaluate.ts`; this file is just I/O + the CLI.
 */

import { readFileSync, readdirSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { TransformersEmbedder } from "../src/core/embedder";
import {
  DEFAULT_CHUNK_OVERLAP,
  DEFAULT_CHUNK_TOKENS,
  DEFAULT_EMBEDDING_MODEL,
  DEFAULT_HYBRID_ALPHA,
  EMBEDDING_MODELS,
} from "../src/core/config";
import type { NoteInput } from "../src/core/types";
import {
  buildIndex,
  CUTOFF,
  evaluateAllModes,
  groundingCheck,
  type GroundingResult,
  type MetricRow,
  type Qrel,
  type WikiQaEntry,
} from "./evaluate";

const DEMO_VAULT_DIR = "demo-vault";
const EVAL_DIR = "eval";
const RESULTS_DIR = join(EVAL_DIR, "results");
const SEMANTIC_POOL = 50;

/** Parse a demo-vault note file into a NoteInput keyed by its filename. */
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

function readJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as T);
}

function printSummary(ranking: Record<string, MetricRow>, grounding: GroundingResult): void {
  console.log("\n=== Retrieval quality (SciFact, @10) ===");
  console.log("mode      nDCG@10   recall@10");
  for (const [mode, row] of Object.entries(ranking)) {
    console.log(`${mode.padEnd(9)} ${row.ndcg.toFixed(4)}    ${row.recall.toFixed(4)}`);
  }
  console.log("\n=== Answer grounding (WikiQA) ===");
  console.log(
    `questions=${grounding.questions}  accuracy@1=${grounding.accuracyAt1.toFixed(4)}  ` +
      `MRR=${grounding.mrr.toFixed(4)}`,
  );
}

async function main(): Promise<void> {
  const modelId = DEFAULT_EMBEDDING_MODEL;
  const dim = EMBEDDING_MODELS[modelId].dim;
  console.log(`Loading embedding model ${modelId}…`);
  const embedder = new TransformersEmbedder({ modelId, dim, useWebGPU: false });

  const files = readdirSync(DEMO_VAULT_DIR).filter((name) => name.endsWith(".md"));
  console.log(`Indexing ${files.length} notes…`);
  const notes = files.map(loadNote);
  const { store, bm25 } = await buildIndex(embedder, notes, {
    chunkTokens: DEFAULT_CHUNK_TOKENS,
    chunkOverlap: DEFAULT_CHUNK_OVERLAP,
  });
  console.log(`Indexed ${store.size} chunks.`);

  const qrels = readJsonl<Qrel>(join(EVAL_DIR, "scifact_qrels.jsonl"));
  console.log(`Scoring ${qrels.length} queries…`);
  const ranking = await evaluateAllModes(
    embedder,
    store,
    bm25,
    qrels,
    DEFAULT_HYBRID_ALPHA,
    SEMANTIC_POOL,
  );

  const wikiqa = readJsonl<WikiQaEntry>(join(EVAL_DIR, "wikiqa_slice.jsonl"));
  console.log(`Grounding check over ${wikiqa.length} WikiQA questions…`);
  const grounding = await groundingCheck(embedder, wikiqa);

  const metrics = {
    generatedAt: new Date().toISOString(),
    model: modelId,
    corpus: { notes: files.length, chunks: store.size },
    queries: qrels.length,
    cutoff: CUTOFF,
    alpha: DEFAULT_HYBRID_ALPHA,
    ranking,
    grounding,
  };

  mkdirSync(RESULTS_DIR, { recursive: true });
  const stamp = metrics.generatedAt.replace(/[:.]/g, "-");
  const outPath = join(RESULTS_DIR, `metrics_${stamp}.json`);
  writeFileSync(outPath, JSON.stringify(metrics, null, 2), "utf8");

  printSummary(ranking, grounding);
  console.log(`\nWrote ${outPath}`);
}

void main();
