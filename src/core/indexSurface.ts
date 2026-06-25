/**
 * The exact text surfaces derived from a chunk for indexing. Shared so the
 * plugin (indexService) and the offline eval embed and lexically index chunks
 * identically — otherwise the eval would measure a different pipeline than ships.
 */

import type { Chunk } from "./types";

/** Text fed to the embedding model: heading-prefixed when the chunk has one. */
export function embedInput(chunk: Chunk): string {
  return chunk.heading.length > 0 ? `${chunk.heading}\n${chunk.text}` : chunk.text;
}

/** Text fed to BM25: note title + heading + chunk text. */
export function lexicalInput(chunk: Chunk): string {
  return `${chunk.noteTitle} ${chunk.heading} ${chunk.text}`.trim();
}
