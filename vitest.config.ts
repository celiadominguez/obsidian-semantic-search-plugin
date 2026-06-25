import { defineConfig } from "vitest/config";

/**
 * The `obsidian` module only exists inside Obsidian, so it is aliased to a local
 * stub for tests. Only `tests/` is collected; `core` and the eval logic are
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
    },
  },
});
