# VaultSeek

VaultSeek is a local-first Obsidian plugin that adds semantic search and cited
Q&A across an entire vault. Embedding, indexing, retrieval, and retrieval-only
Q&A run fully on-device — after a one-time embedding-model download there are no
network calls by default, and the plugin never writes to your notes. It is built
as a practical learning application that explores on-device retrieval-augmented
search inside a desktop editor. For example, a note titled "tapering off
caffeine" becomes findable by searching "reduce coffee", which Obsidian's lexical
search cannot do.

## Architecture

```
vault events (create/modify/delete/rename)
        │  (debounced, content-hash diff)
        ▼
   chunk (heading-aware, ~512 tokens, 64 overlap)
        │
        ▼
   embed  ──►  Web Worker (transformers.js, WebGPU → WASM fallback)
        │
        ▼
   store  ──►  Float32 vectors (exact cosine; HNSW above a threshold)
        │      + BM25 lexical index
        ▼
   search / Q&A  ──►  hybrid rank (α·cosine + (1−α)·bm25)  ──►  cited answer
```

The codebase is split into two layers with a strict rule:

- **`src/core/`** — pure retrieval logic (chunker, embedder, vector store, BM25,
  hybrid ranker, Q&A engine). It imports **nothing from `obsidian`**, so it
  compiles and unit-tests in plain Node. This is what makes the retrieval
  behaviour testable and the eval reproducible.
- **`src/obsidian/`** — thin glue to the Obsidian API (the sidebar view, the
  settings tab, and the vault-event wiring). It is excluded from unit tests.
- **`src/worker/`** — the embedding Web Worker and its main-thread bridge, so the
  UI thread never blocks during indexing.

## Setup

Requirements: **Node.js 22 LTS** and npm.

```bash
npm ci          # install pinned dependencies
npm run build   # bundle src/main.ts → main.js
```

To test it in a vault, copy `main.js`, `manifest.json`, and `styles.css` into
`<your-vault>/.obsidian/plugins/vaultseek/`, then enable VaultSeek in
Obsidian's Community Plugins settings. On first index the embedding model
(~33 MB) is downloaded once and cached on disk; everything after that is offline.

## Configuration

Every setting has a default, so a fresh install runs fully offline with no setup.

| Setting                                           | Default                                  | Purpose                                                    |
| ------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------- |
| `embeddingModel`                                  | `Xenova/bge-small-en-v1.5`               | On-device embedding model (also `Xenova/all-MiniLM-L6-v2`) |
| `useWebGPU`                                       | `true` (auto-fallback to WASM)           | Acceleration backend                                       |
| `chunkTokens` / `chunkOverlap`                    | `512` / `64`                             | Chunk size and overlap (approx. tokens)                    |
| `hybridAlpha`                                     | `0.6`                                    | Semantic vs lexical blend (1.0 = semantic only)            |
| `hnswThreshold`                                   | `20000`                                  | Chunk count above which HNSW replaces exact cosine         |
| `generationBackend`                               | `none`                                   | `none` \| `ollama` \| `hosted`                             |
| `ollamaEndpoint` / `ollamaModel`                  | `http://localhost:11434` / `llama3.1:8b` | Local generation                                           |
| `hostedEndpoint` / `hostedModel` / `hostedApiKey` | empty                                    | Opt-in hosted generation only                              |
| `excludedFolders`                                 | `[]`                                     | Vault folders to skip when indexing                        |

## Usage

- **Index** — indexing starts automatically when the plugin loads. Progress is
  shown in the status bar; edits re-index incrementally (only changed chunks are
  re-embedded). Use the command **VaultSeek: Re-index vault** to rebuild.
- **Search** — open the view from the ribbon or the command palette
  (**VaultSeek: Open semantic search**), type a query, and get ranked results
  with a score, a snippet, and actions to open in a split, insert a link, or copy
  a citation.
- **Q&A** — press Enter or **Ask** to get a grounded answer with inline
  `[[note]]` citations. With the default `none` backend the answer is extractive
  (the most relevant passages, fully offline); `ollama`/`hosted` backends
  synthesize an answer from the retrieved chunks. The engine refuses when the
  best match is below the confidence floor rather than guessing.
- **Index management** — re-index, view stats, and delete the index from the
  command palette and settings.
- **Evaluate** — run `npm run eval` to reproduce the retrieval-quality numbers
  below. It indexes the committed demo vault headlessly and writes a metrics
  JSON to `eval/results/`.

## Results

Measured by `npm run eval` over the committed demo vault (1,000 SciFact notes →
1,010 chunks; 300 SciFact test queries; `α = 0.6`) with the default
`Xenova/bge-small-en-v1.5` model on the WASM backend.

| Ranking          | nDCG@10 | recall@10 |
| ---------------- | ------- | --------- |
| Semantic         | 0.8423  | 0.9210    |
| Lexical (BM25)   | 0.7784  | 0.8597    |
| Hybrid (α = 0.6) | 0.8171  | 0.9036    |

Answer-grounding sanity check (WikiQA slice, 150 questions): the question's
nearest candidate sentence is a correct answer 53.3% of the time (accuracy@1),
with MRR 0.711.

**Honest verdict.** Hybrid beats pure lexical on both nDCG@10 (0.8171 vs 0.7784)
and recall@10 (0.9036 vs 0.8597). On this corpus, however, **pure semantic
ranking is the strongest** — SciFact is a scientific claim-verification task
where meaning matching dominates and the lexical signal adds some noise at the
default blend. The default `α = 0.6` is kept as a general-purpose setting that is
robust across vaults rather than tuned to this benchmark; raising `α` favours
SciFact specifically. These numbers were not tuned against the qrels.

## Privacy

- **Offline by default.** After the one-time embedding-model download, the
  default configuration (`generationBackend: none`) makes **zero network calls**.
  Indexing, search, and retrieval-only Q&A are entirely on-device.
- **Read-only over your vault.** The plugin only ever writes to its own
  `.obsidian/plugins/vaultseek/` data folder (the vector blob and its sidecar).
  It never modifies your notes.
- **Opt-in network paths.** `ollama` sends retrieved chunks to a local server;
  `hosted` sends retrieved chunks to a user-configured endpoint with a
  user-supplied key. In both cases **only the retrieved chunks are sent — never
  the whole vault or the index**.
- **Key-sync caveat.** A hosted API key is stored in the plugin's settings. If
  you sync your `.obsidian` folder across devices, that key syncs with it; treat
  it accordingly.

## Limitations

- **Desktop-only.** `isDesktopOnly` is set; mobile is out of scope for this
  build.
- **Vault-size ceiling.** Vectors are held in memory. Exact cosine is used up to
  `hnswThreshold` chunks and HNSW above it; if the WASM HNSW module fails to
  load, the store falls back to exact cosine, which is slower on very large
  vaults. The practical target is single-user vaults up to roughly tens of
  thousands of notes.
- **Demo-vault and eval provenance.** The committed `demo-vault/` notes are
  derived from the **BeIR/SciFact** corpus (one note per abstract; license
  CC BY-NC 2.0 per the SciFact dataset card). The answer-grounding slice in
  `eval/wikiqa_slice.jsonl` is derived from **WikiQA** (Microsoft Research Data
  License per the WikiQA dataset card). `scripts/build_demo_vault.ts` documents
  exactly how these assets were generated; its output is committed, so a clean
  clone needs no network to build, test, or run.

## License

MIT.
