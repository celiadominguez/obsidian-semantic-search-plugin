/**
 * Hybrid ranking: blend on-device semantic similarity with lexical BM25.
 *
 * `score = alpha * cosine + (1 - alpha) * bm25_norm`, the standard convex blend.
 * Cosine is already bounded and comparable across queries, so it is used
 * directly; BM25 scores are query-dependent in magnitude, so they are min-max
 * normalized to [0, 1] over the candidate set before blending. A larger semantic
 * pool than `topK` is fetched so lexical-only candidates still receive a real
 * cosine rather than a default of zero.
 */

import type { Bm25Index } from "./bm25";
import type { Embedder, RankingMode, SearchResult } from "./types";
import type { VectorStore } from "./vectorStore";

/** How many characters of chunk text to show as a preview snippet. */
const SNIPPET_LENGTH = 240;

/** Multiplier applied to `topK` to size the semantic candidate pool. */
const POOL_FACTOR = 5;

/** Minimum semantic candidate pool, so small `topK` queries still see context. */
const MIN_POOL = 50;

/** Min-max normalize a score map into [0, 1]. A flat map maps everything to 1. */
export function normalizeScores(scores: Map<string, number>): Map<string, number> {
  const values = [...scores.values()];
  if (values.length === 0) {
    return new Map();
  }
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min;
  const normalized = new Map<string, number>();
  for (const [id, value] of scores) {
    normalized.set(id, range === 0 ? 1 : (value - min) / range);
  }
  return normalized;
}

/** One blended candidate before it is resolved to a full {@link SearchResult}. */
export interface BlendedScore {
  id: string;
  score: number;
  semanticScore: number;
  lexicalScore: number;
}

/**
 * Blend semantic and normalized-lexical signals per the hybrid formula.
 *
 * @param semantic - Map of id → raw cosine similarity.
 * @param lexicalNorm - Map of id → BM25 score already normalized to [0, 1].
 * @param alpha - Weight on the semantic term, in [0, 1].
 */
export function blend(
  semantic: Map<string, number>,
  lexicalNorm: Map<string, number>,
  alpha: number,
): BlendedScore[] {
  const ids = new Set<string>([...semantic.keys(), ...lexicalNorm.keys()]);
  const blended: BlendedScore[] = [];
  for (const id of ids) {
    const sem = semantic.get(id) ?? 0;
    const lex = lexicalNorm.get(id) ?? 0;
    blended.push({
      id,
      score: alpha * sem + (1 - alpha) * lex,
      semanticScore: sem,
      lexicalScore: lex,
    });
  }
  blended.sort((a, b) => b.score - a.score);
  return blended;
}

/** Build a short, query-agnostic preview snippet from chunk text. */
function makeSnippet(text: string): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  return collapsed.length <= SNIPPET_LENGTH
    ? collapsed
    : `${collapsed.slice(0, SNIPPET_LENGTH).trimEnd()}…`;
}

/** Inputs to a ranking request. */
export interface RankOptions {
  query: string;
  embedder: Embedder;
  store: VectorStore;
  bm25: Bm25Index;
  alpha: number;
  mode: RankingMode;
  topK: number;
  /**
   * Optional instruction prepended to the query before embedding (used by
   * asymmetric models like BGE). Lexical (BM25) matching always uses the raw
   * query, so the instruction never leaks into term matching.
   */
  queryInstruction?: string;
}

/**
 * Rank chunks for a query under the requested mode.
 *
 * @returns Up to `topK` results ordered by the mode's score, highest first.
 */
export async function rank(options: RankOptions): Promise<SearchResult[]> {
  const { query, embedder, store, bm25, alpha, mode, topK, queryInstruction } = options;
  const poolSize = Math.max(topK * POOL_FACTOR, MIN_POOL);

  const semantic = new Map<string, number>();
  if (mode !== "lexical") {
    const embeddedQuery = `${queryInstruction ?? ""}${query}`;
    const [queryVector] = await embedder.embed([embeddedQuery]);
    for (const hit of await store.search(queryVector, poolSize)) {
      semantic.set(hit.id, hit.score);
    }
  }

  const lexicalRaw = mode === "semantic" ? new Map<string, number>() : bm25.scoreAll(query);
  const lexicalNorm = normalizeScores(lexicalRaw);

  let blended: BlendedScore[];
  if (mode === "semantic") {
    blended = blend(semantic, new Map(), 1);
  } else if (mode === "lexical") {
    blended = blend(new Map(), lexicalNorm, 0);
  } else {
    blended = blend(semantic, lexicalNorm, alpha);
  }

  const results: SearchResult[] = [];
  for (const candidate of blended) {
    const chunk = store.getChunk(candidate.id);
    if (chunk === undefined) {
      continue;
    }
    results.push({
      chunk,
      score: candidate.score,
      semanticScore: candidate.semanticScore,
      lexicalScore: candidate.lexicalScore,
      snippet: makeSnippet(chunk.text),
    });
    if (results.length >= topK) {
      break;
    }
  }
  return results;
}
