/**
 * Reusable, side-effect-free evaluation logic shared by the eval CLI and the
 * acceptance tests. No file or network access happens here: callers pass in an
 * embedder and already-loaded data, so tests can run it offline with the
 * deterministic embedder while `run_eval.ts` runs it with the real model.
 */

import { chunkNote } from "../src/core/chunker";
import { hashText } from "../src/core/hash";
import { cosineSimilarity } from "../src/core/embedder";
import { embedInput, lexicalInput } from "../src/core/indexSurface";
import { Bm25Index } from "../src/core/bm25";
import { VectorStore } from "../src/core/vectorStore";
import { rank } from "../src/core/hybridRanker";
import { DEFAULT_HNSW_THRESHOLD } from "../src/core/config";
import type { Embedder, NoteInput, RankingMode } from "../src/core/types";

/** Rank cutoff used for all reported metrics. */
export const CUTOFF = 10;

/** Single relevance judgement: a query and its relevant note paths. */
export interface Qrel {
  query_id: string;
  query: string;
  relevant: string[];
}

/** One WikiQA grounding item. */
export interface WikiQaEntry {
  id: string;
  question: string;
  answer: string;
  answers: string[];
  candidates: string[];
}

export interface MetricRow {
  ndcg: number;
  recall: number;
}

export interface GroundingResult {
  questions: number;
  accuracyAt1: number;
  mrr: number;
}

export interface IndexBuildOptions {
  chunkTokens: number;
  chunkOverlap: number;
  hnswThreshold?: number;
}

/** Build a vector store and BM25 index over the given notes. */
export async function buildIndex(
  embedder: Embedder,
  notes: NoteInput[],
  options: IndexBuildOptions,
): Promise<{ store: VectorStore; bm25: Bm25Index }> {
  const store = new VectorStore({
    dim: embedder.dim,
    modelId: embedder.modelId,
    hnswThreshold: options.hnswThreshold ?? DEFAULT_HNSW_THRESHOLD,
  });
  const bm25 = new Bm25Index();
  for (const note of notes) {
    const chunks = chunkNote(note, options.chunkTokens, options.chunkOverlap);
    const vectors = await embedder.embed(chunks.map(embedInput));
    chunks.forEach((chunk, i) => {
      store.upsert({ chunk, hash: hashText(chunk.text), vector: vectors[i] });
      bm25.add(chunk.id, lexicalInput(chunk));
    });
  }
  return { store, bm25 };
}

/** nDCG@k for binary relevance over an ordered list of note paths. */
export function ndcgAtK(ranked: string[], relevant: Set<string>, k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, ranked.length); i++) {
    if (relevant.has(ranked[i])) {
      dcg += 1 / Math.log2(i + 2);
    }
  }
  let idcg = 0;
  for (let i = 0; i < Math.min(k, relevant.size); i++) {
    idcg += 1 / Math.log2(i + 2);
  }
  return idcg === 0 ? 0 : dcg / idcg;
}

/** recall@k: fraction of relevant notes within the top k. */
export function recallAtK(ranked: string[], relevant: Set<string>, k: number): number {
  if (relevant.size === 0) {
    return 0;
  }
  const top = new Set(ranked.slice(0, k));
  let hits = 0;
  for (const id of relevant) {
    if (top.has(id)) {
      hits++;
    }
  }
  return hits / relevant.size;
}

/** Collapse chunk-level results to a ranked list of unique note paths. */
export function toNoteRanking(results: { chunk: { notePath: string } }[]): string[] {
  const seen = new Set<string>();
  const ranked: string[] = [];
  for (const result of results) {
    if (!seen.has(result.chunk.notePath)) {
      seen.add(result.chunk.notePath);
      ranked.push(result.chunk.notePath);
    }
  }
  return ranked;
}

/** Score one ranking mode across all qrels, averaging nDCG@10 and recall@10. */
export async function evaluateRanking(
  embedder: Embedder,
  store: VectorStore,
  bm25: Bm25Index,
  qrels: Qrel[],
  mode: RankingMode,
  alpha: number,
  pool: number,
  queryInstruction = "",
): Promise<MetricRow> {
  let ndcg = 0;
  let recall = 0;
  for (const qrel of qrels) {
    const relevant = new Set(qrel.relevant);
    const results = await rank({
      query: qrel.query,
      embedder,
      store,
      bm25,
      alpha,
      mode,
      topK: pool,
      queryInstruction,
    });
    const ranked = toNoteRanking(results);
    ndcg += ndcgAtK(ranked, relevant, CUTOFF);
    recall += recallAtK(ranked, relevant, CUTOFF);
  }
  const n = qrels.length || 1;
  return { ndcg: ndcg / n, recall: recall / n };
}

/** Evaluate all three ranking modes. */
export async function evaluateAllModes(
  embedder: Embedder,
  store: VectorStore,
  bm25: Bm25Index,
  qrels: Qrel[],
  alpha: number,
  pool: number,
  queryInstruction = "",
): Promise<Record<RankingMode, MetricRow>> {
  const modes: RankingMode[] = ["semantic", "lexical", "hybrid"];
  const ranking = {} as Record<RankingMode, MetricRow>;
  for (const mode of modes) {
    ranking[mode] = await evaluateRanking(
      embedder,
      store,
      bm25,
      qrels,
      mode,
      alpha,
      pool,
      queryInstruction,
    );
  }
  return ranking;
}

/** Answer-grounding sanity check: does each question's nearest candidate match? */
export async function groundingCheck(
  embedder: Embedder,
  entries: WikiQaEntry[],
  queryInstruction = "",
): Promise<GroundingResult> {
  let correct = 0;
  let reciprocalRankSum = 0;
  for (const entry of entries) {
    const positives = new Set(entry.answers);
    const [questionVec] = await embedder.embed([`${queryInstruction}${entry.question}`]);
    const candidateVecs = await embedder.embed(entry.candidates);
    const ranked = entry.candidates
      .map((candidate, i) => ({
        candidate,
        score: cosineSimilarity(questionVec, candidateVecs[i]),
      }))
      .sort((a, b) => b.score - a.score);
    if (ranked.length > 0 && positives.has(ranked[0].candidate)) {
      correct++;
    }
    const firstRelevant = ranked.findIndex((r) => positives.has(r.candidate));
    if (firstRelevant >= 0) {
      reciprocalRankSum += 1 / (firstRelevant + 1);
    }
  }
  const n = entries.length || 1;
  return { questions: entries.length, accuracyAt1: correct / n, mrr: reciprocalRankSum / n };
}
