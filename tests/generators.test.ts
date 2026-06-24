import { afterEach, describe, expect, it, vi } from "vitest";
import {
  HostedGenerator,
  LmStudioGenerator,
  createGenerator,
  listOllamaModels,
  listOpenAiModels,
} from "../src/core/qa";
import { defaultSettings } from "../src/core/config";
import type { GenerationRequest } from "../src/core/types";

const REQUEST: GenerationRequest = { question: "q", context: [], prompt: "PROMPT" };

function stubFetch(handler: (url: string, init?: RequestInit) => unknown): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const body = handler(url, init);
      return { ok: true, status: 200, statusText: "OK", json: async () => body } as Response;
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createGenerator", () => {
  it("selects the LM Studio generator for the lmstudio backend", () => {
    const settings = { ...defaultSettings(), generationBackend: "lmstudio" as const };
    const generator = createGenerator(settings);
    expect(generator.id).toBe("lmstudio");
  });
});

describe("LmStudioGenerator", () => {
  it("posts an OpenAI-compatible chat completion to the local server", async () => {
    let calledUrl = "";
    let sentBody: { model?: string; messages?: Array<{ content?: string }> } = {};
    stubFetch((url, init) => {
      calledUrl = url;
      sentBody = JSON.parse(String(init?.body));
      return { choices: [{ message: { content: "grounded answer" } }] };
    });

    const generator = new LmStudioGenerator("http://localhost:1234/v1", "qwen2.5-7b");
    const answer = await generator.generate(REQUEST);

    expect(calledUrl).toBe("http://localhost:1234/v1/chat/completions");
    expect(sentBody.model).toBe("qwen2.5-7b");
    expect(sentBody.messages?.[0].content).toBe("PROMPT");
    expect(answer).toBe("grounded answer");
  });

  it("normalizes a trailing slash on the endpoint", async () => {
    let calledUrl = "";
    stubFetch((url) => {
      calledUrl = url;
      return { choices: [{ message: { content: "" } }] };
    });
    await new LmStudioGenerator("http://localhost:1234/v1/", "m").generate(REQUEST);
    expect(calledUrl).toBe("http://localhost:1234/v1/chat/completions");
  });
});

describe("HostedGenerator", () => {
  it("sends a Bearer token and returns the message content", async () => {
    let auth: string | undefined;
    stubFetch((_url, init) => {
      auth = (init?.headers as Record<string, string>).Authorization;
      return { choices: [{ message: { content: "hi" } }] };
    });
    const answer = await new HostedGenerator(
      "https://api.example/v1/chat",
      "key123",
      "gpt",
    ).generate(REQUEST);
    expect(auth).toBe("Bearer key123");
    expect(answer).toBe("hi");
  });
});

describe("model listing", () => {
  it("lists LM Studio / OpenAI models from /models", async () => {
    let calledUrl = "";
    stubFetch((url) => {
      calledUrl = url;
      return { data: [{ id: "qwen2.5-7b" }, { id: "llama-3.2-3b" }] };
    });
    const models = await listOpenAiModels("http://localhost:1234/v1");
    expect(calledUrl).toBe("http://localhost:1234/v1/models");
    expect(models).toEqual(["qwen2.5-7b", "llama-3.2-3b"]);
  });

  it("lists Ollama models from /api/tags", async () => {
    stubFetch(() => ({ models: [{ name: "llama3.1:8b" }, { name: "mistral" }] }));
    const models = await listOllamaModels("http://localhost:11434");
    expect(models).toEqual(["llama3.1:8b", "mistral"]);
  });
});
