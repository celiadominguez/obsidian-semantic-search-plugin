/**
 * Small helpers for turning vault note paths into display/wikilink names.
 * Lives in its own module so both the search view and the Q&A engine can share
 * it without depending on each other.
 */

/** Strip the folder and `.md` extension to get a wikilink-friendly note name. */
export function noteBasename(notePath: string): string {
  const segment = notePath.split("/").pop() ?? notePath;
  return segment.replace(/\.md$/i, "");
}
