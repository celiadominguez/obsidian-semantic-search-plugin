# VaultSeek

VaultSeek is a local-first Obsidian plugin that adds semantic search across an
entire vault. Embedding, indexing, and retrieval run fully on-device — after a
one-time embedding-model download there are no network calls, and the plugin
never writes to your notes.

## Architecture

```
vault events → chunk (heading-aware) → embed (Web Worker, transformers.js)
            → store (Float32 vectors + BM25) → hybrid rank → ranked results
```

The code is split into a pure `src/core/` layer (chunker, embedder, vector
store, BM25, hybrid ranker) that imports nothing from `obsidian` and is unit
tested in Node, and a thin `src/obsidian/` glue layer (the sidebar view and
vault-event wiring). Embedding runs in a Web Worker so the UI never blocks.

## Setup

Requires Node.js 22 LTS.

```bash
npm ci
npm run build   # bundles src/main.ts → main.js
```

Copy `main.js`, `manifest.json`, and `styles.css` into
`<your-vault>/.obsidian/plugins/vaultseek/` and enable the plugin.

## Usage

Open the search view from the ribbon or the command palette and type a query to
get ranked results with scores, snippets, and open-in-split. Run `npm run eval`
to reproduce the retrieval-quality metrics (nDCG@10 / recall@10 for semantic,
lexical, and hybrid ranking) over the committed demo vault.

## Privacy

The default configuration makes zero network calls after the one-time model
download. The plugin is read-only over your vault and writes only to its own
plugin data folder.

## Provenance

The committed `demo-vault/` notes are derived from the BeIR/SciFact corpus (one
note per abstract). `scripts/build_demo_vault.ts` documents the conversion; its
output is committed so a clean clone needs no network.

## License

MIT.
