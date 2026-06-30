# Beta-testing VaultSleuth

VaultSleuth isn't in the Community Plugins store yet. You can install the latest
release directly from this repo using **BRAT** (Beta Reviewer's Auto-update Tool),
which also keeps it updated as new releases land.

## Install

1. **Install BRAT** — in Obsidian: Settings → Community plugins → **Browse** →
   search **"BRAT"** → Install → Enable.
2. **Add VaultSleuth** — open the command palette (`Ctrl/Cmd+P`) and run
   **"BRAT: Add a beta plugin for testing"**.
3. Paste this repository:

   ```
   celiadominguez/obsidian-semantic-search-plugin
   ```

   Then choose **Add Plugin**. BRAT downloads the latest release and installs it.

4. **Enable** VaultSleuth in Settings → Community plugins.

> To stay on a specific release, use BRAT's "Add a beta plugin with frozen
> version" and enter a tag (e.g. `1.1.1`). Otherwise BRAT auto-updates you to the
> newest release on startup.

## First run

- On first index, the embedding model (~33 MB) downloads once from Hugging Face
  and is cached; a status-bar notice announces it. After that, indexing and
  search are fully offline.
- Open the panel from the **brain-circuit ribbon icon** or via the command
  palette (**VaultSleuth: Open semantic search** / **Open chat**).
- **Desktop only** — VaultSleuth does not load on Obsidian mobile.
- **Chat is optional and off by default.** To enable it, set a generation backend
  (Ollama / LM Studio / hosted) in the plugin settings.

## What's useful to test

- Search quality — does meaning-based search surface the right notes?
- Incremental indexing — edit/create/delete/rename notes and confirm results
  update (watch the status bar).
- The "still indexing" banner on a fresh, large vault.
- Excluded folders, chunk-size/overlap, and the hybrid-alpha slider.
- Chat (if you run a local model) — grounded answers and `[[note]]` citations.
- Advanced: the offline **Local model folder** option (see the README).

## Reporting issues

Please open an issue:
https://github.com/celiadominguez/obsidian-semantic-search-plugin/issues/new/choose

Include your OS, Obsidian version, VaultSleuth version, and any errors from the
developer console (`Ctrl/Cmd+Shift+I`). The bug-report template prompts for these.
