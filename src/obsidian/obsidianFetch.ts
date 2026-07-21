/**
 * A `fetch`-compatible function backed by Obsidian's `requestUrl`.
 *
 * transformers.js downloads model files through `env.fetch`, which defaults to
 * the renderer's global `fetch`. In Obsidian's Electron renderer a direct
 * `fetch` to `huggingface.co` is blocked by the Content-Security-Policy / CORS
 * (the plugin's origin is `app://obsidian.md`), so the model download fails with
 * "Failed to load resource". `requestUrl` issues the request from the main
 * process and is not subject to those restrictions — the same reason the local
 * LLM backends use it. We hand this to `env.fetch` so model downloads succeed.
 */

import { requestUrl } from "obsidian";

/** Minimal init shape transformers.js passes (a `Headers` object under the hood). */
interface FetchInit {
  method?: string;
  headers?: Headers | Record<string, string>;
}

export async function obsidianFetch(input: unknown, init?: unknown): Promise<Response> {
  const url = String(input);
  const opts = (init ?? {}) as FetchInit;

  const headers: Record<string, string> = {};
  if (opts.headers !== undefined) {
    new Headers(opts.headers as HeadersInit).forEach((value, key) => {
      headers[key] = value;
    });
  }

  const response = await requestUrl({
    url,
    method: opts.method ?? "GET",
    headers,
    // Return 4xx/5xx instead of throwing so transformers.js can handle optional
    // files that legitimately 404 (network-level failures still reject).
    throw: false,
  });

  return new Response(response.arrayBuffer, {
    status: response.status,
    headers: response.headers,
  });
}
