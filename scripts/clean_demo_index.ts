/**
 * Remove the persisted VaultSleuth index from the committed demo vault.
 *
 * The index (`index.bin` + `index.json`) is regenerated on the next plugin load,
 * so deleting it forces a clean full re-index — handy for testing the indexing
 * UX or clearing a stale/half-written index. Only the index artifacts are
 * touched; the installed plugin files (main.js, manifest.json, styles.css) and
 * every note in the vault are left untouched.
 */

import { rmSync } from "node:fs";
import { join } from "node:path";

const PLUGIN_DIR = join("demo-vault", ".obsidian", "plugins", "vaultsleuth");
const INDEX_FILES = ["index.bin", "index.json"];

let removed = 0;
for (const file of INDEX_FILES) {
  const path = join(PLUGIN_DIR, file);
  // force:true makes a missing file a no-op rather than an error.
  rmSync(path, { force: true });
  console.log(`cleared ${path}`);
  removed++;
}

console.log(`Done — ${removed} index file(s) cleared. The plugin will rebuild on next load.`);
