# VaultSeek

VaultSeek is a local-first Obsidian plugin that adds semantic search, cited Q&A,
and a vault-grounded chat panel across an entire vault. Embedding, indexing,
retrieval, and retrieval-only Q&A/chat run fully on-device — after a one-time
embedding-model download there are no
network calls by default, and the plugin never writes to your notes. It is built
as a practical learning application that explores on-device retrieval-augmented
search inside a desktop editor. For example, a note titled "tapering off
caffeine" becomes findable by searching "reduce coffee", which Obsidian's lexical
search cannot do.

## How it works (in plain terms)

Think of VaultSeek as a very patient **librarian** who has quietly read every note
you've ever written. Instead of remembering exact words, the librarian remembers
what each note _means_. So when you ask for "how to cut back on coffee," they walk
straight to your note about "tapering off caffeine" — even though it doesn't use
any of the same words. Everything the librarian does happens **on your own
computer**: your notes never leave the building.

```mermaid
flowchart TD
    A["📓 Your notes<br/>(stay on your computer)"] --> B["✂️ Split each note into<br/>bite-size passages"]
    B --> C["🧠 Turn each passage into numbers<br/>that capture its meaning<br/>— done on your machine"]
    C --> D["🗂️ A private index,<br/>saved on your computer"]

    Q["❓ You type a question<br/>or search"] --> E["🧠 Turn your question into<br/>the same kind of numbers"]
    E --> F{"🔎 Find the passages<br/>closest in meaning"}
    D --> F

    F --> G["📋 A ranked list of matching notes<br/>you can click to open"]
    F --> H["💬 Optional: a local AI writes a short<br/>answer that points back to your notes,<br/>and says 'I don't know' if it's unsure"]

    classDef onDevice fill:#e8f0fe,stroke:#4285f4,color:#202124;
    class A,B,C,D,E,F,G,H onDevice;
```

**The pieces, in everyday words:**

- **Your notes** — the Markdown files in your vault. VaultSeek only ever _reads_
  them.
- **Splitter** — long notes are cut into small passages so matches can be precise.
- **Meaning-maker (the embedding model)** — a small AI that runs on your computer
  and turns text into a list of numbers representing its meaning. Similar meaning →
  similar numbers.
- **The index** — a private filing cabinet of those numbers, stored in the
  plugin's own folder.
- **Search** — your question gets the same number treatment, and VaultSeek finds
  the passages whose numbers are closest.
- **Chat (optional)** — if you connect a local AI (Ollama or LM Studio), it reads
  the matched passages and writes a short, cited answer. By default this step is
  off and you just get the matching passages.

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
| `generationBackend`                               | `none`                                   | `none` \| `ollama` \| `lmstudio` \| `hosted`               |
| `ollamaEndpoint` / `ollamaModel`                  | `http://localhost:11434` / `llama3.1:8b` | Local generation (Ollama)                                  |
| `lmstudioEndpoint` / `lmstudioModel`              | `http://localhost:1234/v1` / _(picked)_  | Local generation (LM Studio, OpenAI-compatible)            |
| `hostedEndpoint` / `hostedModel` / `hostedApiKey` | empty                                    | Opt-in hosted generation only                              |
| `excludedFolders`                                 | `[]`                                     | Vault folders to skip when indexing                        |

## Usage

- **Index** — indexing starts automatically when the plugin loads. Progress is
  shown in the status bar; edits re-index incrementally (only changed chunks are
  re-embedded). Use the command **VaultSeek: Re-index vault** to rebuild.
- **Search & Chat** — open the panel from the ribbon or the command palette
  (**VaultSeek: Open semantic search** / **Open chat**). One view with a
  **Search ⇄ Chat** toggle sharing a single input box:
  - **Search** gives ranked results with a score, a snippet, and actions to open
    in a split, insert a link, or copy a citation.
  - **Chat** is a multi-turn conversation grounded in your vault: each message
    runs a fresh hybrid retrieval, prior turns are threaded into a
    prompt-injection-safe prompt, and replies carry `[[note]]` citations. With
    the default `none` backend replies are extractive (most relevant passages,
    fully offline); select a local (`ollama` or `lmstudio`) or `hosted` backend in
    settings for conversational answers — the model picker lists the models your
    local server has available. **New chat** clears the conversation. The engine
    refuses when retrieval is below the confidence floor rather than guessing.
- **Index management** — re-index, view stats, and delete the index from the
  command palette and settings.
- **Evaluate** — run `npm run eval` to reproduce the retrieval-quality numbers
  below. It indexes the committed demo vault headlessly and writes a metrics
  JSON to `eval/results/`.

## Results

Measured by `npm run eval` over the committed demo vault (1,000 SciFact notes →
1,010 chunks; 300 SciFact test queries; `α = 0.6`) with the default
`Xenova/bge-small-en-v1.5` model. Queries are embedded with the model's
recommended instruction prefix (BGE is an asymmetric retriever); passages are
not.

| Ranking          | nDCG@10 | recall@10 |
| ---------------- | ------- | --------- |
| Semantic         | 0.8361  | 0.9243    |
| Lexical (BM25)   | 0.7784  | 0.8597    |
| Hybrid (α = 0.6) | 0.8147  | 0.8986    |

Answer-grounding sanity check (WikiQA slice, 150 questions): the question's
nearest candidate sentence is a correct answer 58.0% of the time (accuracy@1),
with MRR 0.730.

**Honest verdict.** Hybrid beats pure lexical on both nDCG@10 (0.8147 vs 0.7784)
and recall@10 (0.8986 vs 0.8597). On this corpus, however, **pure semantic
ranking is the strongest** — SciFact is a scientific claim-verification task
where meaning matching dominates and the lexical signal adds some noise at the
default blend. The default `α = 0.6` is kept as a general-purpose setting that is
robust across vaults rather than tuned to this benchmark; raising `α` favours
SciFact specifically. These numbers were not tuned against the qrels.

**A note on refusal.** Cited Q&A and chat refuse when the best retrieved chunk
falls below a per-model cosine floor. Dense embeddings are anisotropic — even
unrelated text has a non-trivial baseline cosine — so the floor is calibrated per
model (0.5 for BGE) and the query instruction prefix is what cleanly separates
genuine queries from out-of-domain noise. The floor is deliberately biased toward
catching noise, so an occasional weakly-matching real question is declined rather
than answered from thin context.

## Privacy

- **Offline by default.** After the one-time embedding-model download, the
  default configuration (`generationBackend: none`) makes **zero network calls**.
  Indexing, search, and retrieval-only Q&A are entirely on-device.
- **Read-only over your vault.** The plugin only ever writes to its own
  `.obsidian/plugins/vaultseek/` data folder (the vector blob and its sidecar).
  It never modifies your notes.
- **Opt-in network paths.** `ollama` and `lmstudio` send retrieved chunks to a
  local server (localhost, on-device); `hosted` sends retrieved chunks to a
  user-configured endpoint with a user-supplied key. In all cases **only the
  retrieved chunks are sent — never the whole vault or the index**.
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
