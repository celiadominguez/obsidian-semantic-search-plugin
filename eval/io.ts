/**
 * Shared file I/O for the offline tooling (eval CLI and demo-vault generator).
 *
 * These helpers are deliberately tiny and Node-only; they never run inside the
 * plugin. Keeping them in one place avoids the slightly-divergent copies that
 * previously lived in `run_eval.ts` and `build_demo_vault.ts`.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { NoteInput } from "../src/core/types";

/** Read a newline-delimited JSON file into an array, skipping blank lines. */
export function readJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as T);
}

/** Parse a demo-vault note file into a {@link NoteInput} keyed by its filename. */
export function loadNote(dir: string, fileName: string): NoteInput {
  const raw = readFileSync(join(dir, fileName), "utf8");
  const titleMatch = /title:\s*"([^"]+)"/.exec(raw);
  return {
    path: fileName,
    title: titleMatch?.[1] ?? fileName.replace(/\.md$/, ""),
    content: raw,
    mtime: 0,
  };
}
