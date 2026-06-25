/**
 * Tiny, dependency-free string hashing used for content-based chunk diffing.
 *
 * FNV-1a is chosen over a crypto hash because the goal is change detection, not
 * security: it is fast, deterministic, and identical across Node and the browser
 * (no `node:crypto` dependency), which keeps `core` portable.
 */

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

/** Compute a stable 32-bit FNV-1a hash of `text`, returned as 8 hex chars. */
export function hashText(text: string): string {
  let hash = FNV_OFFSET_BASIS;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, FNV_PRIME);
  }
  // Coerce to unsigned 32-bit and pad to a fixed width for stable comparison.
  return (hash >>> 0).toString(16).padStart(8, "0");
}
