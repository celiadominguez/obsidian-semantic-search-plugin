# Changelog

All notable changes to VaultSleuth are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-07-22

### Fixed

- Keep the UI responsive during a full index rebuild. On-device embedding runs on
  the main thread, so a large rebuild could make the app (including closing the
  settings window) feel frozen. The indexing loops now yield to the event loop on
  a time slice so the interface keeps painting and accepting input.

### Changed

- Clarified in settings that you can close the window during a rebuild — indexing
  continues in the background, with progress in the status bar.

## [1.2.0] - 2026-07-21

### Added

- **Remote embedding models (opt-in).** Embeddings can now be computed by an
  external server instead of the bundled on-device model. Supports
  OpenAI-compatible servers (LM Studio, Text Embeddings Inference, Infinity,
  hosted APIs) and Ollama's native endpoint, with a **Detect** button that reads
  the model's vector dimension, an optional API key, and a configurable query
  instruction for asymmetric models (e.g. BGE). On-device remains the default.
- **Chat backend readiness check.** Opening chat now warns up front if the
  configured local model isn't running or isn't loaded, instead of only failing
  after a message is sent.
- **Rebuild index** button and a live **Index status** (notes / chunks) readout
  in the settings' Maintenance section.
- **Refresh** button next to the Ollama / LM Studio model dropdowns to re-query a
  server without reopening settings.

### Changed

- **Reorganized the settings tab** by how often each option is used: Search &
  indexing, Chat, Advanced (tuning), and Maintenance. Moved "Excluded folders"
  next to the indexing settings and gave the chat-backend dropdown readable
  labels.
- Replaced the unresolved ribbon/view icon with a valid one.
- Documented the reorganized settings and remote-embeddings setup in the README.

### Fixed

- Switching embedding server, protocol, model, or dimension now correctly
  invalidates the persisted index so stale vectors are never mixed with new ones.

## [1.1.1] - 2026-06-25

### Fixed

- Normalize user-entered paths and read the offline local model through the Vault
  API.

## [1.1.0] - 2026-06-25

### Added

- Vault-grounded chat with cited answers; local (Ollama / LM Studio) and opt-in
  hosted generation backends.

## [1.0.0] - 2026-06-25

### Added

- Initial release: on-device semantic + hybrid (BM25) search over an Obsidian
  vault.

[1.2.1]: https://github.com/celiadominguez/obsidian-semantic-search-plugin/releases/tag/1.2.1
[1.2.0]: https://github.com/celiadominguez/obsidian-semantic-search-plugin/releases/tag/1.2.0
[1.1.1]: https://github.com/celiadominguez/obsidian-semantic-search-plugin/releases/tag/1.1.1
[1.1.0]: https://github.com/celiadominguez/obsidian-semantic-search-plugin/releases/tag/1.1.0
[1.0.0]: https://github.com/celiadominguez/obsidian-semantic-search-plugin/releases/tag/1.0.0
