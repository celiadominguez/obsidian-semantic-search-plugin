import { defineConfig } from "vitest/config";

/**
 * The `obsidian` module and the build-time `inline:embed-worker` virtual module
 * only exist inside Obsidian / the esbuild bundle, so they are aliased to local
 * stubs for tests. Only `tests/` is collected; `core` and the eval logic are
 * exercised directly without ever loading Obsidian.
 */
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
  resolve: {
    alias: {
      obsidian: new URL("./tests/mocks/obsidian.ts", import.meta.url).pathname,
      "inline:embed-worker": new URL("./tests/mocks/inlineWorker.ts", import.meta.url).pathname,
    },
  },
});
