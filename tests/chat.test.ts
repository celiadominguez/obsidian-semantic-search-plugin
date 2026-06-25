import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { ChatEngine, buildChatPrompt, renderHistory } from "../src/core/chat";
import { HashingEmbedder } from "../src/core/embedder";
import { NoneGenerator } from "../src/core/generation";
import { defaultSettings } from "../src/core/config";
import { buildIndex } from "../eval/evaluate";
import type { ChatMessage, Generator, NoteInput } from "../src/core/types";

const DEMO_VAULT_DIR = "demo-vault";
const DIM = 384;

function loadSubset(size = 60): NoteInput[] {
  return readdirSync(DEMO_VAULT_DIR)
    .filter((name) => name.endsWith(".md"))
    .sort()
    .slice(0, size)
    .map((fileName) => {
      const raw = readFileSync(join(DEMO_VAULT_DIR, fileName), "utf8");
      const titleMatch = /title:\s*"([^"]+)"/.exec(raw);
      return {
        path: fileName,
        title: titleMatch?.[1] ?? fileName.replace(/\.md$/, ""),
        content: raw,
        mtime: 0,
      };
    });
}

async function makeEngine(floor = 0.1) {
  const notes = loadSubset();
  const embedder = new HashingEmbedder(DIM);
  const { chunkTokens, chunkOverlap } = defaultSettings();
  const { store, bm25 } = await buildIndex(embedder, notes, { chunkTokens, chunkOverlap });
  const engine = new ChatEngine({
    embedder,
    store,
    bm25,
    generator: new NoneGenerator(),
    alpha: 0.6,
    similarityFloor: floor,
  });
  return { engine, notes };
}

describe("renderHistory", () => {
  it("is empty for no history", () => {
    expect(renderHistory([])).toBe("");
  });

  it("renders user and assistant turns inside a delimiter", () => {
    const history: ChatMessage[] = [
      { role: "user", content: "hello", citations: [], refused: false },
      { role: "assistant", content: "hi", citations: [], refused: false },
    ];
    const rendered = renderHistory(history);
    expect(rendered).toContain("<conversation>");
    expect(rendered).toContain("User: hello");
    expect(rendered).toContain("Assistant: hi");
  });
});

describe("buildChatPrompt", () => {
  it("frames context as untrusted and includes prior history and the question", () => {
    const context = [
      {
        chunk: {
          id: "n.md#0",
          notePath: "n.md",
          noteTitle: "N",
          heading: "",
          ordinal: 0,
          text: "body",
        },
        score: 1,
        semanticScore: 1,
        lexicalScore: 1,
        snippet: "body",
      },
    ];
    const history: ChatMessage[] = [
      { role: "user", content: "earlier", citations: [], refused: false },
    ];
    const prompt = buildChatPrompt("what is it?", context, history);
    expect(prompt).toContain("untrusted data");
    expect(prompt).toContain("<context>");
    expect(prompt).toContain("User: earlier");
    expect(prompt).toContain("Question: what is it?");
  });
});

describe("ChatEngine", () => {
  it("accumulates multi-turn history (user + assistant per turn)", async () => {
    const { engine, notes } = await makeEngine();
    await engine.ask(notes[0].title);
    await engine.ask(notes[1].title);
    expect(engine.history).toHaveLength(4);
    expect(engine.history.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
  });

  it("grounds replies with citations that resolve to real notes", async () => {
    const { engine, notes } = await makeEngine();
    const reply = await engine.ask(notes[0].title);
    expect(reply.message.refused).toBe(false);
    expect(reply.message.grounded).toBe(true);
    expect(reply.message.citations.length).toBeGreaterThan(0);
    for (const citation of reply.message.citations) {
      expect(notes.some((note) => note.path === citation)).toBe(true);
    }
  });

  it("refuses on weak retrieval with the offline (none) backend", async () => {
    const { engine } = await makeEngine(0.99);
    const reply = await engine.ask("qwxz vbnmqp plokju ytrewq zxcvbn asdfgh");
    expect(reply.message.refused).toBe(true);
    expect(reply.message.grounded).toBe(false);
    expect(reply.message.citations).toHaveLength(0);
    expect(engine.history).toHaveLength(2);
  });

  it("answers via the model on weak retrieval when a generative backend is set", async () => {
    const notes = loadSubset();
    const embedder = new HashingEmbedder(DIM);
    const { chunkTokens, chunkOverlap } = defaultSettings();
    const { store, bm25 } = await buildIndex(embedder, notes, { chunkTokens, chunkOverlap });
    // A stand-in generative backend; a high floor forces the weak-match path.
    const llm: Generator = { id: "ollama", generate: async () => "An answer from the model." };
    const engine = new ChatEngine({
      embedder,
      store,
      bm25,
      generator: llm,
      alpha: 0.6,
      similarityFloor: 0.99,
    });
    const reply = await engine.ask("a question the notes do not cover");
    expect(reply.message.refused).toBe(false);
    expect(reply.message.grounded).toBe(false);
    expect(reply.message.content).toContain("An answer from the model.");
  });

  it("reset clears the conversation", async () => {
    const { engine, notes } = await makeEngine();
    await engine.ask(notes[0].title);
    expect(engine.history.length).toBeGreaterThan(0);
    engine.reset();
    expect(engine.history).toHaveLength(0);
  });
});
