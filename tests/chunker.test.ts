import { describe, expect, it } from "vitest";
import { chunkNote, stripFrontmatter } from "../src/core/chunker";
import type { NoteInput } from "../src/core/types";

function note(content: string, path = "note.md", title = "Note"): NoteInput {
  return { path, title, content, mtime: 0 };
}

describe("stripFrontmatter", () => {
  it("removes a leading YAML frontmatter block", () => {
    const stripped = stripFrontmatter("---\ntitle: x\nid: 1\n---\nBody text");
    expect(stripped).toBe("Body text");
  });

  it("leaves content without frontmatter untouched", () => {
    expect(stripFrontmatter("No frontmatter here")).toBe("No frontmatter here");
  });

  it("strips an empty frontmatter block", () => {
    expect(stripFrontmatter("---\n---\nBody")).toBe("Body");
  });

  it("does not strip a leading thematic break around prose", () => {
    // A note that opens with a `---` horizontal rule, prose, then another rule
    // must not be mistaken for YAML frontmatter and silently deleted.
    const content = "---\nThis is a real paragraph of prose, not YAML.\n---\nMore body";
    expect(stripFrontmatter(content)).toBe(content);
  });

  it("strips frontmatter that mixes keys and list items", () => {
    const stripped = stripFrontmatter("---\ntags:\n  - a\n  - b\ntitle: x\n---\nBody");
    expect(stripped).toBe("Body");
  });
});

describe("chunkNote", () => {
  it("creates one chunk per short heading section with correct headings", () => {
    const content = "# Alpha\nfirst section body\n\n## Beta\nsecond section body";
    const chunks = chunkNote(note(content), 512, 64);
    expect(chunks).toHaveLength(2);
    expect(chunks[0].heading).toBe("Alpha");
    expect(chunks[1].heading).toBe("Beta");
    expect(chunks.map((c) => c.ordinal)).toEqual([0, 1]);
    expect(chunks.map((c) => c.id)).toEqual(["note.md#0", "note.md#1"]);
  });

  it("indexes a heading-only note (e.g. a Map-of-Content) by its headings", () => {
    const content = "# Index\n## Projects\n## Reading list";
    const chunks = chunkNote(note(content), 512, 64);
    expect(chunks.map((c) => c.text)).toEqual(["Index", "Projects", "Reading list"]);
  });

  it("strips frontmatter before chunking", () => {
    const content = "---\ntitle: T\n---\n# H\nhello world";
    const chunks = chunkNote(note(content), 512, 64);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].text).toBe("hello world");
    expect(chunks[0].text).not.toContain("title");
  });

  it("windows a long section with the configured size and overlap", () => {
    const words = Array.from({ length: 250 }, (_, i) => `w${i}`);
    const content = `# Long\n${words.join(" ")}`;
    const size = 100;
    const overlap = 20;
    const chunks = chunkNote(note(content), size, overlap);

    // Step is size - overlap = 80, so windows start at 0, 80, 160 (the third
    // window reaches the end and the loop stops).
    expect(chunks).toHaveLength(3);
    for (const chunk of chunks) {
      expect(chunk.text.split(" ").length).toBeLessThanOrEqual(size);
      expect(chunk.heading).toBe("Long");
    }

    // Adjacent windows must share exactly `overlap` tokens.
    const first = chunks[0].text.split(" ");
    const second = chunks[1].text.split(" ");
    expect(first.slice(size - overlap)).toEqual(second.slice(0, overlap));
  });

  it("produces contiguous ordinals across multiple sections", () => {
    const longBody = Array.from({ length: 150 }, (_, i) => `t${i}`).join(" ");
    const content = `# One\n${longBody}\n\n# Two\nshort`;
    const chunks = chunkNote(note(content), 100, 10);
    expect(chunks.map((c) => c.ordinal)).toEqual(chunks.map((_, i) => i));
    expect(chunks[chunks.length - 1].heading).toBe("Two");
  });
});
