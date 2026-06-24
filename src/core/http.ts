/**
 * Minimal HTTP transport abstraction for the generation backends.
 *
 * Why this exists: in Obsidian's renderer a browser `fetch()` to a local server
 * (Ollama, LM Studio) is blocked by CORS — the plugin's origin is `app://obsidian.md`
 * and those servers send no `Access-Control-Allow-Origin` header. Obsidian's
 * `requestUrl()` issues the request from the main process and bypasses CORS, but
 * it lives in the `obsidian` module, which `core` must not import. So `core`
 * depends only on this tiny interface; the Obsidian layer injects a
 * `requestUrl`-backed client, while Node and tests use the `fetch` default.
 */

/** The subset of an HTTP response the backends need. */
export interface HttpResponse {
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
}

/** A pluggable HTTP client. */
export type HttpClient = (
  url: string,
  init?: { method?: string; headers?: Record<string, string>; body?: string },
) => Promise<HttpResponse>;

/** Default client backed by the global `fetch` (Node, tests, non-CORS contexts). */
export const defaultHttpClient: HttpClient = async (url, init) => {
  const response = await fetch(url, init);
  return {
    ok: response.ok,
    status: response.status,
    json: () => response.json(),
  };
};
