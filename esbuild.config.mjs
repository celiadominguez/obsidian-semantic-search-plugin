/**
 * esbuild bundler for the VaultSeek Obsidian plugin.
 *
 * Why this shape: Obsidian loads a single CommonJS `main.js` from the plugin
 * folder, so the entire plugin (including the embedding Web Worker) must be
 * bundled into that one artifact. The worker is bundled separately into an
 * in-memory string and exposed through the virtual module `inline:embed-worker`;
 * at runtime the main thread instantiates it from a Blob URL, which keeps the
 * plugin a single shippable file with no sidecar scripts to resolve on disk.
 */

import esbuild from "esbuild";
import process from "node:process";

const PRODUCTION = process.argv.includes("production");

const BANNER =
  "/* VaultSeek — local-first semantic search for Obsidian. Bundled artifact; edit sources under src/. */";

// Modules provided by the Obsidian runtime or Node — never bundle these.
const EXTERNALS = [
  "obsidian",
  "electron",
  "@codemirror/autocomplete",
  "@codemirror/collab",
  "@codemirror/commands",
  "@codemirror/language",
  "@codemirror/lint",
  "@codemirror/search",
  "@codemirror/state",
  "@codemirror/view",
  "@lezer/common",
  "@lezer/highlight",
  "@lezer/lr",
  "node:fs",
  "node:path",
  "node:os",
  "node:crypto",
];

/**
 * esbuild plugin that bundles the embedding worker into a string and serves it
 * through the `inline:embed-worker` import specifier.
 */
const inlineWorkerPlugin = {
  name: "inline-embed-worker",
  setup(build) {
    build.onResolve({ filter: /^inline:embed-worker$/ }, () => ({
      path: "embed-worker",
      namespace: "inline-worker",
    }));

    build.onLoad({ filter: /.*/, namespace: "inline-worker" }, async () => {
      const result = await esbuild.build({
        entryPoints: ["src/worker/embedWorker.ts"],
        bundle: true,
        format: "iife",
        platform: "browser",
        target: "es2022",
        minify: PRODUCTION,
        write: false,
        sourcemap: false,
        external: ["onnxruntime-node"],
      });
      const code = result.outputFiles[0].text;
      return {
        contents: `export default ${JSON.stringify(code)};`,
        loader: "js",
      };
    });
  },
};

const buildOptions = {
  entryPoints: ["src/main.ts"],
  bundle: true,
  format: "cjs",
  platform: "browser",
  target: "es2022",
  external: EXTERNALS,
  banner: { js: BANNER },
  sourcemap: PRODUCTION ? false : "inline",
  treeShaking: true,
  minify: PRODUCTION,
  outfile: "main.js",
  logLevel: "info",
  plugins: [inlineWorkerPlugin],
};

if (PRODUCTION) {
  await esbuild.build(buildOptions);
} else {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log("VaultSeek: esbuild watching for changes…");
}
