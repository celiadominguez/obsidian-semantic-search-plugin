/**
 * Heading-aware Markdown chunker.
 *
 * Why heading-aware: embedding a whole note dilutes topical signal, while a
 * fixed-size sliding window cuts across unrelated sections. Splitting on Markdown
 * headings first keeps each chunk topically coherent; only sections longer than
 * the token budget are then windowed (with overlap) so no chunk exceeds the
 * embedding model's context and adjacent windows share boundary context.
 *
 * "Tokens" here are approximated by whitespace-delimited words. This is
 * intentionally simple and model-agnostic; the embedding model truncates at its
 * own limit, and the approximation only needs to keep chunks comfortably within
 * that bound while producing stable, testable boundaries.
 */

import type { Chunk, NoteInput } from "./types";

const FRONTMATTER_RE = /^---\r?\n[\s\S]*?\r?\n---\r?\n?/;
const HEADING_RE = /^(#{1,6})\s+(.*)$/;

/** Remove a leading YAML frontmatter block, if present. */
export function stripFrontmatter(content: string): string {
  return content.replace(FRONTMATTER_RE, "");
}

interface Section {
  heading: string;
  body: string;
}

/** Split note body into sections delimited by Markdown ATX headings. */
function splitIntoSections(content: string): Section[] {
  const lines = content.split("\n");
  const sections: Section[] = [];
  let heading = "";
  let buffer: string[] = [];

  const flush = (): void => {
    const body = buffer.join("\n").trim();
    if (body.length > 0) {
      sections.push({ heading, body });
    }
    buffer = [];
  };

  for (const line of lines) {
    const match = HEADING_RE.exec(line);
    if (match) {
      flush();
      heading = match[2].trim();
    } else {
      buffer.push(line);
    }
  }
  flush();
  return sections;
}

/** Tokenize on whitespace, preserving the words for re-joining. */
function tokenize(text: string): string[] {
  return text.split(/\s+/).filter((token) => token.length > 0);
}

/**
 * Window a token array into overlapping slices.
 *
 * @returns Arrays of tokens, each at most `size` long, advancing by
 *   `size - overlap` tokens per window so consecutive windows share `overlap`.
 */
function windowTokens(tokens: string[], size: number, overlap: number): string[][] {
  if (tokens.length <= size) {
    return [tokens];
  }
  const step = Math.max(1, size - overlap);
  const windows: string[][] = [];
  for (let start = 0; start < tokens.length; start += step) {
    windows.push(tokens.slice(start, start + size));
    if (start + size >= tokens.length) {
      break;
    }
  }
  return windows;
}

/**
 * Chunk a note into heading-aware, token-bounded slices.
 *
 * @param note - The note to chunk.
 * @param chunkTokens - Maximum approximate tokens per chunk.
 * @param chunkOverlap - Token overlap between adjacent windows of one section.
 * @returns Ordered chunks; `ordinal` is contiguous across the whole note.
 */
export function chunkNote(note: NoteInput, chunkTokens: number, chunkOverlap: number): Chunk[] {
  const overlap = Math.min(chunkOverlap, Math.max(0, chunkTokens - 1));
  const body = stripFrontmatter(note.content);
  const sections = splitIntoSections(body);
  const chunks: Chunk[] = [];

  for (const section of sections) {
    const tokens = tokenize(section.body);
    if (tokens.length === 0) {
      continue;
    }
    for (const window of windowTokens(tokens, chunkTokens, overlap)) {
      const ordinal = chunks.length;
      chunks.push({
        id: `${note.path}#${ordinal}`,
        notePath: note.path,
        noteTitle: note.title,
        heading: section.heading,
        ordinal,
        text: window.join(" "),
      });
    }
  }

  return chunks;
}
