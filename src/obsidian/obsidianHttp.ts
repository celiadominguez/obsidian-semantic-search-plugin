/**
 * An {@link HttpClient} backed by Obsidian's `requestUrl`, which issues requests
 * from the main process and is therefore not subject to the renderer's CORS
 * policy. Local servers like Ollama and LM Studio do not send CORS headers, so a
 * plain `fetch` from the plugin's `app://obsidian.md` origin is blocked; routing
 * through `requestUrl` makes those local backends reachable.
 */

import { requestUrl } from "obsidian";
import type { HttpClient } from "../core/http";

export const obsidianHttpClient: HttpClient = async (url, init) => {
  const response = await requestUrl({
    url,
    method: init?.method ?? "GET",
    headers: init?.headers,
    body: init?.body,
    // Return the response instead of throwing on 4xx/5xx; callers inspect status.
    throw: false,
  });
  return {
    ok: response.status >= 200 && response.status < 300,
    status: response.status,
    // `requestUrl` exposes the parsed body as a property, not a method.
    json: async () => response.json,
  };
};
