/**
 * Compact in-repo BM25 lexical index.
 *
 * Why hand-rolled: the only lexical signal the hybrid ranker needs is a per-chunk
 * BM25 score, so pulling in a full search-engine dependency would be overkill.
 * The index supports incremental add/remove because the vault is indexed
 * incrementally as notes change. IDF is computed at query time from the live
 * document frequencies so scores stay correct as documents come and go.
 */

import { BM25_B, BM25_K1 } from "./config";

/** A scored document id from a BM25 query. */
export interface LexicalHit {
  id: string;
  score: number;
}

const TOKEN_RE = /[a-z0-9]+/g;

/** Lowercase, alphanumeric word tokenization shared with the query path. */
export function tokenizeForBm25(text: string): string[] {
  return text.toLowerCase().match(TOKEN_RE) ?? [];
}

export class Bm25Index {
  private readonly docFreq = new Map<string, number>();
  private readonly docTermFreq = new Map<string, Map<string, number>>();
  private readonly docLength = new Map<string, number>();
  private totalLength = 0;

  /** Number of indexed documents. */
  public get size(): number {
    return this.docTermFreq.size;
  }

  /** Mean document length, used for BM25 length normalization. */
  private get averageLength(): number {
    return this.size === 0 ? 0 : this.totalLength / this.size;
  }

  /** Add or replace a document. Re-adding an existing id updates it cleanly. */
  public add(id: string, text: string): void {
    if (this.docTermFreq.has(id)) {
      this.remove(id);
    }
    const tokens = tokenizeForBm25(text);
    const termFreq = new Map<string, number>();
    for (const token of tokens) {
      termFreq.set(token, (termFreq.get(token) ?? 0) + 1);
    }
    for (const term of termFreq.keys()) {
      this.docFreq.set(term, (this.docFreq.get(term) ?? 0) + 1);
    }
    this.docTermFreq.set(id, termFreq);
    this.docLength.set(id, tokens.length);
    this.totalLength += tokens.length;
  }

  /** Remove a document by id. No-op if the id is unknown. */
  public remove(id: string): void {
    const termFreq = this.docTermFreq.get(id);
    if (termFreq === undefined) {
      return;
    }
    for (const term of termFreq.keys()) {
      const next = (this.docFreq.get(term) ?? 0) - 1;
      if (next <= 0) {
        this.docFreq.delete(term);
      } else {
        this.docFreq.set(term, next);
      }
    }
    this.totalLength -= this.docLength.get(id) ?? 0;
    this.docTermFreq.delete(id);
    this.docLength.delete(id);
  }

  /** Remove every document. */
  public clear(): void {
    this.docFreq.clear();
    this.docTermFreq.clear();
    this.docLength.clear();
    this.totalLength = 0;
  }

  /** Robertson–Sparck-Jones IDF with the usual +1 to keep it non-negative. */
  private idf(term: string): number {
    const df = this.docFreq.get(term) ?? 0;
    if (df === 0) {
      return 0;
    }
    return Math.log(1 + (this.size - df + 0.5) / (df + 0.5));
  }

  /**
   * Score every document containing at least one query term.
   *
   * @returns A map of document id to raw (un-normalized) BM25 score.
   */
  public scoreAll(query: string): Map<string, number> {
    const scores = new Map<string, number>();
    const queryTerms = new Set(tokenizeForBm25(query));
    const avgdl = this.averageLength;
    if (avgdl === 0) {
      return scores;
    }
    for (const term of queryTerms) {
      const idf = this.idf(term);
      if (idf === 0) {
        continue;
      }
      for (const [id, termFreq] of this.docTermFreq) {
        const tf = termFreq.get(term);
        if (tf === undefined) {
          continue;
        }
        const length = this.docLength.get(id) ?? 0;
        const denominator = tf + BM25_K1 * (1 - BM25_B + (BM25_B * length) / avgdl);
        const contribution = idf * ((tf * (BM25_K1 + 1)) / denominator);
        scores.set(id, (scores.get(id) ?? 0) + contribution);
      }
    }
    return scores;
  }

  /** Rank documents by BM25 score, highest first. */
  public search(query: string, topK: number): LexicalHit[] {
    const scores = this.scoreAll(query);
    return [...scores.entries()]
      .map(([id, score]) => ({ id, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}
