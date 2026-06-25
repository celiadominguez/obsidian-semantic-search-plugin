/**
 * Generation backends plus the shared answer helpers (context rendering and
 * `[[note]]` citation binding) used by the chat engine.
 *
 * Four backends share one interface:
 *  - `none` (default): fully offline, extractive — the answer is built directly
 *    from the retrieved chunks, with no text generation and no network.
 *  - `ollama` / `lmstudio`: a local generation server (opt-in, localhost).
 *  - `hosted`: an OpenAI-compatible endpoint (opt-in, user key).
 *
 * In every non-`none` case only the retrieved chunks are sent — never the whole
 * vault or the index. Retrieved content is wrapped in explicit `<context>`
 * delimiters and framed as untrusted data by the caller's prompt, so a note
 * containing adversarial text like "ignore previous instructions" is treated as
 * data, not as a command.
 */

import { defaultHttpClient, type HttpClient } from "./http";
import { noteBasename } from "./notePath";
import type { GenerationRequest, Generator, SearchResult, VaultSleuthSettings } from "./types";

/** Returned verbatim when retrieval is too weak to answer confidently. */
export const REFUSAL_MESSAGE =
  "No confident answer: your vault does not contain enough relevant material to answer this question.";

const WIKILINK_RE = /\[\[([^\]]+)\]\]/g;

/**
 * Render retrieved chunks as a delimited, injection-safe context block. Each
 * chunk is numbered and tagged with its note name; callers wrap this in
 * `<context>` tags and frame it as untrusted data.
 */
export function renderContextBlock(context: SearchResult[]): string {
  return context
    .map((result, i) => {
      const tag = `[${i + 1}] note: ${noteBasename(result.chunk.notePath)}`;
      return `${tag}\n${result.chunk.text}`;
    })
    .join("\n---\n");
}

/** Extract the note names referenced by `[[...]]` in answer text. */
function extractWikilinks(text: string): string[] {
  const names: string[] = [];
  for (const match of text.matchAll(WIKILINK_RE)) {
    names.push(match[1].split("|")[0].split("#")[0].trim());
  }
  return names;
}

/**
 * Resolve the citations in an answer to real note paths drawn from the context.
 * Citations that do not match any context note are dropped, so the result only
 * ever points at notes that actually exist.
 */
export function resolveCitations(answer: string, context: SearchResult[]): string[] {
  const byBasename = new Map<string, string>();
  for (const result of context) {
    byBasename.set(noteBasename(result.chunk.notePath).toLowerCase(), result.chunk.notePath);
  }
  const resolved: string[] = [];
  const seen = new Set<string>();
  for (const name of extractWikilinks(answer)) {
    const path = byBasename.get(name.toLowerCase());
    if (path !== undefined && !seen.has(path)) {
      seen.add(path);
      resolved.push(path);
    }
  }
  return resolved;
}

/** Offline, extractive generator: the answer is the grounding context itself. */
export class NoneGenerator implements Generator {
  public readonly id = "none" as const;

  public async generate(request: GenerationRequest): Promise<string> {
    const bullets = request.context.map((result) => {
      return `- ${result.snippet} ([[${noteBasename(result.chunk.notePath)}]])`;
    });
    return ["Retrieval-only mode — most relevant passages from your vault:", "", ...bullets].join(
      "\n",
    );
  }
}

/** Drop a trailing slash so endpoints concatenate cleanly. */
function trimSlash(url: string): string {
  return url.replace(/\/$/, "");
}

/** Local Ollama generator (opt-in, localhost). */
export class OllamaGenerator implements Generator {
  public readonly id = "ollama" as const;

  constructor(
    private readonly endpoint: string,
    private readonly model: string,
    private readonly http: HttpClient = defaultHttpClient,
  ) {}

  public async generate(request: GenerationRequest): Promise<string> {
    const response = await this.http(`${trimSlash(this.endpoint)}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, prompt: request.prompt, stream: false }),
    });
    if (!response.ok) {
      throw new Error(`Ollama request failed: ${response.status}`);
    }
    const data = (await response.json()) as { response?: string };
    return data.response ?? "";
  }
}

/**
 * POST an OpenAI-compatible chat completion and return the message content.
 * Shared by the hosted and LM Studio backends, which differ only in auth.
 */
async function openAiChatCompletion(
  http: HttpClient,
  url: string,
  model: string,
  prompt: string,
  apiKey?: string,
): Promise<string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey !== undefined && apiKey.length > 0) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  const response = await http(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
    }),
  });
  if (!response.ok) {
    throw new Error(`Generation failed: ${response.status}`);
  }
  const data = (await response.json()) as { choices?: Array<{ message?: { content?: string } }> };
  return data.choices?.[0]?.message?.content ?? "";
}

/** List model ids from an OpenAI-compatible `/models` endpoint (e.g. LM Studio). */
export async function listOpenAiModels(
  baseUrl: string,
  apiKey?: string,
  http: HttpClient = defaultHttpClient,
): Promise<string[]> {
  const headers: Record<string, string> = {};
  if (apiKey !== undefined && apiKey.length > 0) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  const response = await http(`${trimSlash(baseUrl)}/models`, { headers });
  if (!response.ok) {
    throw new Error(`Listing models failed: ${response.status}`);
  }
  const data = (await response.json()) as { data?: Array<{ id?: string }> };
  return (data.data ?? []).map((m) => m.id).filter((id): id is string => typeof id === "string");
}

/** List installed Ollama model names from its `/api/tags` endpoint. */
export async function listOllamaModels(
  endpoint: string,
  http: HttpClient = defaultHttpClient,
): Promise<string[]> {
  const response = await http(`${trimSlash(endpoint)}/api/tags`);
  if (!response.ok) {
    throw new Error(`Listing models failed: ${response.status}`);
  }
  const data = (await response.json()) as { models?: Array<{ name?: string }> };
  return (data.models ?? []).map((m) => m.name).filter((n): n is string => typeof n === "string");
}

/**
 * Local LM Studio generator (opt-in). LM Studio exposes an OpenAI-compatible
 * server on localhost, so only retrieved chunks are sent and nothing leaves the
 * machine.
 */
export class LmStudioGenerator implements Generator {
  public readonly id = "lmstudio" as const;

  constructor(
    private readonly endpoint: string,
    private readonly model: string,
    private readonly http: HttpClient = defaultHttpClient,
  ) {}

  public async generate(request: GenerationRequest): Promise<string> {
    return openAiChatCompletion(
      this.http,
      `${trimSlash(this.endpoint)}/chat/completions`,
      this.model,
      request.prompt,
    );
  }
}

/** Hosted, OpenAI-compatible generator (opt-in, user key; sends only chunks). */
export class HostedGenerator implements Generator {
  public readonly id = "hosted" as const;

  constructor(
    private readonly endpoint: string,
    private readonly apiKey: string,
    private readonly model: string,
    private readonly http: HttpClient = defaultHttpClient,
  ) {}

  public async generate(request: GenerationRequest): Promise<string> {
    return openAiChatCompletion(this.http, this.endpoint, this.model, request.prompt, this.apiKey);
  }
}

/** Build the generator selected by the user's settings, using the given HTTP client. */
export function createGenerator(
  settings: VaultSleuthSettings,
  http: HttpClient = defaultHttpClient,
): Generator {
  switch (settings.generationBackend) {
    case "ollama":
      return new OllamaGenerator(settings.ollamaEndpoint, settings.ollamaModel, http);
    case "lmstudio":
      return new LmStudioGenerator(settings.lmstudioEndpoint, settings.lmstudioModel, http);
    case "hosted":
      return new HostedGenerator(
        settings.hostedEndpoint,
        settings.hostedApiKey,
        settings.hostedModel,
        http,
      );
    case "none":
    default:
      return new NoneGenerator();
  }
}
