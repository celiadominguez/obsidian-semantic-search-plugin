import { describe, expect, it } from "vitest";
import {
  HostedGenerator,
  LmStudioGenerator,
  OllamaGenerator,
  createGenerator,
  listOllamaModels,
  listOpenAiModels,
} from "../src/core/generation";
import { defaultSettings } from "../src/core/config";
import type { HttpClient } from "../src/core/http";
import type { GenerationRequest } from "../src/core/types";

const REQUEST: GenerationRequest = { question: "q", context: [], prompt: "PROMPT" };

/** One recorded request: the URL plus the parsed JSON body and headers sent. */
interface RecordedCall {
  url: string;
  body: unknown;
  headers: Record<string, string> | undefined;
}

/**
 * An injected HttpClient stub that records every call and answers with a canned
 * JSON body. Generators take their transport as a dependency, so tests never
 * touch a global fetch.
 */
function stubHttp(body: unknown, ok = true, status = 200) {
  const calls: RecordedCall[] = [];
  const http: HttpClient = async (url, init) => {
    calls.push({
      url,
      body: init?.body === undefined ? undefined : JSON.parse(init.body),
      headers: init?.headers,
    });
    return { ok, status, json: async () => body };
  };
  return { http, calls };
}

/** A client that answers every request with a non-2xx status. */
function failingHttp(status: number): HttpClient {
  return stubHttp({}, false, status).http;
}

describe("createGenerator", () => {
  it("maps each backend setting to the matching generator", () => {
    const base = defaultSettings();
    const { http } = stubHttp({});
    expect(createGenerator({ ...base, generationBackend: "none" }, http).id).toBe("none");
    expect(createGenerator({ ...base, generationBackend: "ollama" }, http).id).toBe("ollama");
    expect(createGenerator({ ...base, generationBackend: "lmstudio" }, http).id).toBe("lmstudio");
    expect(createGenerator({ ...base, generationBackend: "hosted" }, http).id).toBe("hosted");
  });
});

describe("OllamaGenerator", () => {
  it("posts to /api/generate with stream:false and returns data.response", async () => {
    const { http, calls } = stubHttp({ response: "ollama answer" });
    const answer = await new OllamaGenerator(
      "http://localhost:11434/",
      "llama3.1:8b",
      http,
    ).generate(REQUEST);

    expect(calls[0].url).toBe("http://localhost:11434/api/generate");
    expect(calls[0].body).toEqual({ model: "llama3.1:8b", prompt: "PROMPT", stream: false });
    expect(answer).toBe("ollama answer");
  });
});

describe("generation error handling", () => {
  it("rejects when the backend returns a non-ok status", async () => {
    const http = failingHttp(500);
    await expect(
      new OllamaGenerator("http://localhost:11434", "m", http).generate(REQUEST),
    ).rejects.toThrow(/500/);
    await expect(
      new LmStudioGenerator("http://localhost:1234/v1", "m", http).generate(REQUEST),
    ).rejects.toThrow(/500/);
  });

  it("rejects model listing on a non-ok status", async () => {
    const http = failingHttp(503);
    await expect(listOpenAiModels("http://localhost:1234/v1", undefined, http)).rejects.toThrow(
      /503/,
    );
    await expect(listOllamaModels("http://localhost:11434", http)).rejects.toThrow(/503/);
  });
});

describe("LmStudioGenerator", () => {
  it("posts an OpenAI-compatible chat completion to the local server", async () => {
    const { http, calls } = stubHttp({ choices: [{ message: { content: "grounded answer" } }] });
    const generator = new LmStudioGenerator("http://localhost:1234/v1", "qwen2.5-7b", http);
    const answer = await generator.generate(REQUEST);

    expect(calls[0].url).toBe("http://localhost:1234/v1/chat/completions");
    const body = calls[0].body as { model?: string; messages?: Array<{ content?: string }> };
    expect(body.model).toBe("qwen2.5-7b");
    expect(body.messages?.[0].content).toBe("PROMPT");
    expect(answer).toBe("grounded answer");
  });

  it("normalizes a trailing slash on the endpoint", async () => {
    const { http, calls } = stubHttp({ choices: [{ message: { content: "" } }] });
    await new LmStudioGenerator("http://localhost:1234/v1/", "m", http).generate(REQUEST);
    expect(calls[0].url).toBe("http://localhost:1234/v1/chat/completions");
  });
});

describe("HostedGenerator", () => {
  it("sends a Bearer token and returns the message content", async () => {
    const { http, calls } = stubHttp({ choices: [{ message: { content: "hi" } }] });
    const answer = await new HostedGenerator(
      "https://api.example/v1/chat",
      "key123",
      "gpt",
      http,
    ).generate(REQUEST);

    expect(calls[0].headers?.Authorization).toBe("Bearer key123");
    expect(answer).toBe("hi");
  });
});

describe("model listing", () => {
  it("lists LM Studio / OpenAI models from /models", async () => {
    const { http, calls } = stubHttp({ data: [{ id: "qwen2.5-7b" }, { id: "llama-3.2-3b" }] });
    const models = await listOpenAiModels("http://localhost:1234/v1", undefined, http);
    expect(calls[0].url).toBe("http://localhost:1234/v1/models");
    expect(models).toEqual(["qwen2.5-7b", "llama-3.2-3b"]);
  });

  it("lists Ollama models from /api/tags", async () => {
    const { http } = stubHttp({ models: [{ name: "llama3.1:8b" }, { name: "mistral" }] });
    const models = await listOllamaModels("http://localhost:11434", http);
    expect(models).toEqual(["llama3.1:8b", "mistral"]);
  });
});
