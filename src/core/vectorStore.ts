/**
 * In-memory vector store with exact-cosine search and an optional HNSW index for
 * large vaults, plus a portable (de)serialization format.
 *
 * Design choices:
 *  - Vectors live as `Float32Array`s keyed by chunk id; a secondary note→chunks
 *    map makes note deletion and rename O(chunks-in-note) rather than O(store).
 *  - Below `hnswThreshold` chunks, exact cosine is both fast enough and exactly
 *    correct, so it is always used. Above the threshold the store lazily builds
 *    an HNSW graph (transformers-free, via `hnswlib-wasm`); if that WASM module
 *    fails to load it falls back to exact cosine and the caller can surface the
 *    vault-size caveat. Either way results are well-defined.
 *  - Persistence is split into a compact binary blob (the contiguous vectors)
 *    and a JSON sidecar (chunk metadata + content hashes + index metadata). The
 *    store only (de)serializes to in-memory buffers; actual file I/O is the
 *    caller's job, which keeps this module free of `obsidian` and `node:fs`.
 */

import { cosineSimilarity } from "./embedder";
import type { Chunk, IndexMetadata, VectorRecord } from "./types";

/** On-disk schema version for the sidecar format. */
export const SIDECAR_VERSION = 1;

/** A scored chunk id from a vector search (score is cosine similarity). */
export interface SemanticHit {
  id: string;
  score: number;
}

/** JSON sidecar persisted next to the binary vector blob. */
export interface VectorSidecar {
  meta: IndexMetadata;
  entries: Array<{ chunk: Chunk; hash: string }>;
}

interface VectorStoreOptions {
  dim: number;
  modelId: string;
  hnswThreshold: number;
}

/** Structural view of the parts of the `hnswlib-wasm` index we use. */
interface HnswIndex {
  initIndex(maxElements: number, m: number, efConstruction: number, randomSeed: number): void;
  setEfSearch(ef: number): void;
  addPoint(point: number[], label: number, replaceDeleted: boolean): void;
  searchKnn(
    query: number[],
    k: number,
    filter?: unknown,
  ): { neighbors: number[]; distances: number[] };
}

/** Spare capacity built into a fresh graph so later inserts avoid a rebuild. */
const HNSW_GROWTH = 1.5;
const HNSW_MIN_HEADROOM = 1024;

/** Lazily-loaded HNSW searcher. Isolated so a WASM failure can be caught cleanly. */
class HnswSearcher {
  private readonly index: HnswIndex;
  private readonly labelToId = new Map<number, string>();
  private nextLabel: number;
  private readonly capacity: number;

  private constructor(index: HnswIndex, labelToId: Map<number, string>, capacity: number) {
    this.index = index;
    this.labelToId = labelToId;
    this.nextLabel = labelToId.size;
    this.capacity = capacity;
  }

  /**
   * Build an HNSW graph over the given records, with spare capacity so later
   * single inserts can be added without rebuilding. Returns `null` if
   * `hnswlib-wasm` cannot be loaded or initialized, signalling the caller to use
   * exact cosine.
   */
  public static async build(
    records: VectorRecord[],
    dim: number,
    params: { m: number; efConstruction: number; efSearch: number },
  ): Promise<HnswSearcher | null> {
    try {
      const lib = await import("hnswlib-wasm");
      const module = (await lib.loadHnswlib()) as unknown as {
        HierarchicalNSW: new (space: string, dim: number) => HnswIndex;
      };
      const index = new module.HierarchicalNSW("cosine", dim);
      const capacity = Math.max(
        Math.ceil(records.length * HNSW_GROWTH),
        records.length + HNSW_MIN_HEADROOM,
      );
      index.initIndex(capacity, params.m, params.efConstruction, 200);
      index.setEfSearch(params.efSearch);
      const labelToId = new Map<number, string>();
      records.forEach((record, label) => {
        index.addPoint(Array.from(record.vector), label, false);
        labelToId.set(label, record.chunk.id);
      });
      return new HnswSearcher(index, labelToId, capacity);
    } catch {
      return null;
    }
  }

  /**
   * Add one new point to the existing graph. Returns false when at capacity (the
   * caller then rebuilds). Only valid for genuinely new ids — hnswlib cannot
   * update a label in place.
   */
  public tryAddPoint(record: VectorRecord): boolean {
    if (this.nextLabel >= this.capacity) {
      return false;
    }
    try {
      this.index.addPoint(Array.from(record.vector), this.nextLabel, false);
    } catch {
      return false;
    }
    this.labelToId.set(this.nextLabel, record.chunk.id);
    this.nextLabel++;
    return true;
  }

  public search(query: Float32Array, topK: number): SemanticHit[] {
    const k = Math.min(topK, this.labelToId.size);
    const result = this.index.searchKnn(Array.from(query), k, undefined);
    const hits: SemanticHit[] = [];
    for (let i = 0; i < result.neighbors.length; i++) {
      const id = this.labelToId.get(result.neighbors[i]);
      if (id !== undefined) {
        // hnswlib cosine space returns distance = 1 - cosine similarity.
        hits.push({ id, score: 1 - result.distances[i] });
      }
    }
    return hits;
  }
}

export class VectorStore {
  private readonly records = new Map<string, VectorRecord>();
  private readonly byNote = new Map<string, Set<string>>();
  private readonly dim: number;
  private readonly modelId: string;
  private readonly hnswThreshold: number;
  private hnsw: HnswSearcher | null = null;
  private hnswDirty = true;

  constructor(options: VectorStoreOptions) {
    this.dim = options.dim;
    this.modelId = options.modelId;
    this.hnswThreshold = options.hnswThreshold;
  }

  /** Number of stored vectors. */
  public get size(): number {
    return this.records.size;
  }

  /** True when an HNSW graph (rather than exact cosine) backs search. */
  public get usesHnsw(): boolean {
    return this.size > this.hnswThreshold;
  }

  /** Whether a chunk id is present. */
  public has(id: string): boolean {
    return this.records.has(id);
  }

  /** Chunk metadata for a stored id, or `undefined` if absent. */
  public getChunk(id: string): Chunk | undefined {
    return this.records.get(id)?.chunk;
  }

  /** Map of chunk id → content hash for every chunk of a note. */
  public hashesForNote(notePath: string): Map<string, string> {
    const result = new Map<string, string>();
    for (const id of this.byNote.get(notePath) ?? []) {
      const hash = this.records.get(id)?.hash;
      if (hash !== undefined) {
        result.set(id, hash);
      }
    }
    return result;
  }

  /** Insert or replace a chunk's vector. */
  public upsert(record: VectorRecord): void {
    if (record.vector.length !== this.dim) {
      throw new Error(
        `Vector dimension ${record.vector.length} does not match store dimension ${this.dim}`,
      );
    }
    const isReplace = this.records.has(record.chunk.id);
    this.records.set(record.chunk.id, record);
    let set = this.byNote.get(record.chunk.notePath);
    if (set === undefined) {
      set = new Set<string>();
      this.byNote.set(record.chunk.notePath, set);
    }
    set.add(record.chunk.id);
    // Keep a live HNSW graph current cheaply: a genuinely new chunk is added in
    // place; a replacement (hnswlib can't update a label) forces a later rebuild.
    if (this.hnsw !== null && !this.hnswDirty) {
      if (isReplace || !this.hnsw.tryAddPoint(record)) {
        this.hnswDirty = true;
      }
    } else if (this.hnsw === null) {
      this.hnswDirty = true;
    }
  }

  /** Remove a single chunk by id. */
  public remove(id: string): void {
    const record = this.records.get(id);
    if (record === undefined) {
      return;
    }
    this.records.delete(id);
    const set = this.byNote.get(record.chunk.notePath);
    if (set !== undefined) {
      set.delete(id);
      if (set.size === 0) {
        this.byNote.delete(record.chunk.notePath);
      }
    }
    this.hnswDirty = true;
  }

  /** Remove every chunk belonging to a note (used on delete/rename). */
  public removeNote(notePath: string): void {
    for (const id of [...(this.byNote.get(notePath) ?? [])]) {
      this.remove(id);
    }
  }

  /** Remove everything. */
  public clear(): void {
    this.records.clear();
    this.byNote.clear();
    this.hnsw = null;
    this.hnswDirty = true;
  }

  /**
   * Search for the `topK` most cosine-similar chunks to `query`.
   *
   * Uses exact cosine at or below the HNSW threshold (exact and fast for small
   * vaults), and the HNSW graph above it, transparently falling back to exact
   * cosine if the graph cannot be built.
   */
  public async search(query: Float32Array, topK: number): Promise<SemanticHit[]> {
    if (this.usesHnsw) {
      const hnsw = await this.ensureHnsw();
      if (hnsw !== null) {
        return hnsw.search(query, topK);
      }
    }
    return this.exactSearch(query, topK);
  }

  private async ensureHnsw(): Promise<HnswSearcher | null> {
    if (this.hnsw !== null && !this.hnswDirty) {
      return this.hnsw;
    }
    const { HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH } = await import("./config");
    this.hnsw = await HnswSearcher.build([...this.records.values()], this.dim, {
      m: HNSW_M,
      efConstruction: HNSW_EF_CONSTRUCTION,
      efSearch: HNSW_EF_SEARCH,
    });
    this.hnswDirty = false;
    return this.hnsw;
  }

  /** Brute-force cosine over all vectors; exact top-k. */
  private exactSearch(query: Float32Array, topK: number): SemanticHit[] {
    const hits: SemanticHit[] = [];
    for (const record of this.records.values()) {
      hits.push({ id: record.chunk.id, score: cosineSimilarity(query, record.vector) });
    }
    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, topK);
  }

  /** Current index metadata, for persistence and reload validation. */
  public metadata(): IndexMetadata {
    return { version: SIDECAR_VERSION, modelId: this.modelId, dim: this.dim, count: this.size };
  }

  /**
   * Serialize to a portable representation: a contiguous `Float32` blob of all
   * vectors plus a JSON sidecar describing them. Vector order matches sidecar
   * `entries` order.
   */
  public toBlob(): { vectors: ArrayBuffer; sidecar: VectorSidecar } {
    const entries: VectorSidecar["entries"] = [];
    const blob = new Float32Array(this.size * this.dim);
    let offset = 0;
    for (const record of this.records.values()) {
      blob.set(record.vector, offset);
      offset += this.dim;
      entries.push({ chunk: record.chunk, hash: record.hash });
    }
    return { vectors: blob.buffer, sidecar: { meta: this.metadata(), entries } };
  }

  /** Reconstruct a store from a blob + sidecar produced by {@link toBlob}. */
  public static fromBlob(
    vectors: ArrayBuffer,
    sidecar: VectorSidecar,
    hnswThreshold: number,
  ): VectorStore {
    const { meta, entries } = sidecar;
    const flat = new Float32Array(vectors);
    const expected = entries.length * meta.dim;
    if (flat.length !== expected) {
      throw new Error(
        `Vector blob length ${flat.length} does not match sidecar (${expected} expected)`,
      );
    }
    const store = new VectorStore({ dim: meta.dim, modelId: meta.modelId, hnswThreshold });
    entries.forEach((entry, i) => {
      const start = i * meta.dim;
      const vector = flat.slice(start, start + meta.dim);
      store.upsert({ chunk: entry.chunk, hash: entry.hash, vector });
    });
    return store;
  }
}
