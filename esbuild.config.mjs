/**
 * esbuild bundler for the VaultSleuth Obsidian plugin.
 *
 * Obsidian loads a single CommonJS `main.js` from the plugin folder, so the whole
 * plugin — including the bundled transformers.js embedding model runtime — is
 * emitted into that one artifact alongside `manifest.json` and `styles.css`.
 */

import esbuild from "esbuild";
import process from "node:process";

const PRODUCTION = process.argv.includes("production");

const BANNER =
  "/* VaultSleuth — local-first semantic search for Obsidian. Bundled artifact; edit sources under src/. */";

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
};

if (PRODUCTION) {
  await esbuild.build(buildOptions);
} else {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log("VaultSleuth: esbuild watching for changes…");
}
