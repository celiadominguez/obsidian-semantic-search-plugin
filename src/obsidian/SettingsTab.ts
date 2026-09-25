/**
 * Settings tab exposing every configuration key, built on Obsidian's declarative
 * settings API (`getSettingDefinitions`, 1.13+). Declaring settings rather than
 * rendering them by hand makes every option discoverable through Obsidian's
 * settings search and lets the app own the rendering, persistence, and
 * re-evaluation of conditional rows.
 *
 * Organized by how often a setting is touched, so the everyday knobs come first
 * and the tuning ones stay out of the way:
 *   1. Search & indexing — embedding source/model, excluded folders
 *   2. Chat (answer generation) — backend + its fields
 *   3. Advanced (tuning) — chunking, hybrid blend, HNSW threshold
 *   4. Maintenance — index status + rebuild
 *
 * Simple values are declared as bound controls. Rows that need behaviour the
 * declarative controls cannot express — server-populated model pickers with a
 * Refresh button, the dimension probe, masked secrets, and the rebuild action —
 * are declared as `render` rows, which stay searchable by name but draw their
 * own control. Changing the embedding model re-indexes automatically; chunking
 * and other tuning changes take effect on the next rebuild. Anything that sends
 * data off the machine (hosted generation, remote embeddings) is clearly
 * labelled opt-in.
 */

import {
  Notice,
  normalizePath,
  type Plugin,
  PluginSettingTab,
  type Setting,
  type SettingDefinitionItem,
} from "obsidian";
import {
  EMBEDDING_MODELS,
  GENERATION_BACKENDS,
  MAX_CHUNK_TOKENS,
  MAX_HNSW_THRESHOLD,
} from "../core/config";
import { listOllamaModels, listOpenAiModels } from "../core/generation";
import { detectRemoteEmbeddingDim } from "../core/remoteEmbedder";
import { obsidianHttpClient } from "./obsidianHttp";
import type { IndexStats } from "./indexService";
import type { GenerationBackend, VaultSleuthSettings } from "../core/types";

/** Human-readable labels for the chat backend dropdown (values stay the ids). */
const GENERATION_BACKEND_LABELS: Record<GenerationBackend, string> = {
  none: "None — offline, retrieval only (default)",
  ollama: "Ollama (local server)",
  lmstudio: "LM Studio (local server)",
  hosted: "Hosted API (opt-in, sends chunks out)",
};

/** Keys whose value decides which other rows are shown; a change re-renders the tab. */
const VISIBILITY_KEYS = new Set<string>([
  "embeddingSource",
  "generationBackend",
  "remoteEmbeddingProtocol",
]);

/** Free-text keys trimmed on save (URLs, model names, secrets). */
const TRIMMED_KEYS = new Set<string>([
  "ollamaEndpoint",
  "lmstudioEndpoint",
  "hostedEndpoint",
  "hostedModel",
  "hostedApiKey",
  "remoteEmbeddingEndpoint",
  "remoteEmbeddingApiKey",
]);

/** What the settings tab needs from the plugin, decoupled from the class. */
export interface SettingsHost extends Plugin {
  settings: VaultSleuthSettings;
  saveSettings(): Promise<void>;
  /** Re-embed the whole vault (after a model or chunking change). */
  requestReindex(): Promise<void>;
  /** Current index statistics, for the maintenance status readout. */
  indexStats(): IndexStats;
}

export class SettingsTab extends PluginSettingTab {
  private readonly host: SettingsHost;

  constructor(host: SettingsHost) {
    super(host.app, host);
    this.host = host;
  }

  private get settings(): VaultSleuthSettings {
    return this.host.settings;
  }

  /** Read a bound control's value from the plugin settings. */
  public getControlValue(key: string): unknown {
    // The excluded-folders control edits one comma-separated line; the setting
    // itself is an array of normalized paths.
    if (key === "excludedFolders") {
      return this.settings.excludedFolders.join(", ");
    }
    return (this.settings as unknown as Record<string, unknown>)[key];
  }

  /**
   * Persist a bound control's value, normalizing free text, then run the side
   * effects a key needs: a model change re-indexes, and a change to a key that
   * gates other rows re-renders the tab so those rows appear or disappear.
   */
  public async setControlValue(key: string, value: unknown): Promise<void> {
    const settings = this.settings as unknown as Record<string, unknown>;
    if (key === "excludedFolders") {
      settings.excludedFolders = String(value)
        .split(",")
        .map((folder) => folder.trim())
        .filter((folder) => folder.length > 0)
        .map((folder) => normalizePath(folder));
    } else if (key === "localModelPath") {
      const trimmed = String(value).trim();
      // Keep "" (disabled) as-is — normalizePath turns an empty string into "/",
      // which would look like a real folder.
      settings.localModelPath = trimmed.length > 0 ? normalizePath(trimmed) : "";
    } else if (TRIMMED_KEYS.has(key)) {
      settings[key] = String(value).trim();
    } else {
      settings[key] = value;
    }
    await this.host.saveSettings();
    if (key === "embeddingModel") {
      await this.host.requestReindex();
    }
    if (VISIBILITY_KEYS.has(key)) {
      this.update();
    }
  }

  public getSettingDefinitions(): SettingDefinitionItem[] {
    const onDevice = (): boolean => this.settings.embeddingSource === "on-device";
    const remote = (): boolean => this.settings.embeddingSource === "remote";
    const backend = (id: GenerationBackend) => (): boolean =>
      this.settings.generationBackend === id;

    return [
      {
        type: "group",
        heading: "Search & indexing",
        items: [
          {
            name: "Embedding source",
            desc:
              "Where embeddings are computed. 'On-device' runs the bundled model locally — " +
              "nothing leaves your machine. 'Remote server' delegates to an embeddings API: " +
              "private if you point it at localhost (Ollama / LM Studio), but it sends your " +
              "note text off your machine if you point it at a hosted service. " +
              "Switching clears the index — rebuild it afterwards.",
            control: {
              type: "dropdown",
              key: "embeddingSource",
              options: { "on-device": "On-device (default)", remote: "Remote server" },
            },
          },
          {
            name: "Embedding model",
            desc: "On-device model used to embed notes. Changing it triggers a full re-index.",
            visible: onDevice,
            control: {
              type: "dropdown",
              key: "embeddingModel",
              options: Object.fromEntries(
                Object.entries(EMBEDDING_MODELS).map(([id, info]) => [id, info.label]),
              ),
            },
          },
          {
            name: "Local model folder (offline, advanced)",
            desc:
              "Optional. A vault folder containing the model files, laid out as " +
              "<folder>/<model id>/… (e.g. onnx/model_quantized.onnx, config.json, " +
              "tokenizer.json). When set, the model loads from disk and is never " +
              "downloaded. Leave empty to download it once from Hugging Face. " +
              "Experimental — rebuild the index and confirm search still works after setting it.",
            visible: onDevice,
            control: { type: "text", key: "localModelPath", placeholder: "e.g. models" },
          },
          {
            name: "Server type",
            desc:
              "'OpenAI-compatible' works with LM Studio, Text Embeddings Inference, Infinity, " +
              "and hosted APIs (POST /embeddings). 'Ollama' uses its native POST /api/embed.",
            visible: remote,
            control: {
              type: "dropdown",
              key: "remoteEmbeddingProtocol",
              options: { openai: "OpenAI-compatible", ollama: "Ollama" },
            },
          },
          {
            name: "Embeddings endpoint",
            desc:
              "Base URL of the embeddings server — e.g. http://localhost:1234/v1 for an " +
              "OpenAI-compatible server, or http://localhost:11434 for Ollama.",
            visible: remote,
            control: { type: "text", key: "remoteEmbeddingEndpoint" },
          },
          {
            name: "Embedding model",
            desc: "Pick an embedding model the server offers. Changing it clears the index — rebuild afterwards.",
            visible: remote,
            render: (setting) =>
              this.renderModelDropdown(setting, "remoteEmbeddingModel", () =>
                this.settings.remoteEmbeddingProtocol === "ollama"
                  ? listOllamaModels(this.settings.remoteEmbeddingEndpoint, obsidianHttpClient)
                  : listOpenAiModels(
                      this.settings.remoteEmbeddingEndpoint,
                      this.settings.remoteEmbeddingApiKey,
                      obsidianHttpClient,
                    ),
              ),
          },
          {
            name: "Vector dimension",
            desc: "Must match what the model returns. Use Detect to ask the server, then rebuild the index.",
            visible: remote,
            render: (setting) => this.renderDimension(setting),
          },
          {
            name: "API key (optional)",
            desc:
              "Only needed for servers that require auth. Stored locally and sent " +
              "as a bearer token to the endpoint above, nowhere else.",
            visible: remote,
            render: (setting) => this.renderSecret(setting, "remoteEmbeddingApiKey"),
          },
          {
            name: "Query instruction (advanced)",
            desc:
              "Prepended to your query (not your notes) before embedding, for asymmetric " +
              "models like the BGE family — e.g. 'Represent this sentence for searching " +
              "relevant passages: '. Leave empty for symmetric models. Applies to queries " +
              "only, so no re-index is needed.",
            visible: remote,
            control: {
              type: "text",
              key: "remoteEmbeddingQueryInstruction",
              placeholder: "Represent this sentence for searching relevant passages: ",
            },
          },
          {
            name: "Excluded folders",
            desc:
              "Comma-separated vault paths to skip when indexing. Applies to new edits immediately; " +
              "rebuild the index (below) to drop already-indexed notes.",
            control: { type: "textarea", key: "excludedFolders" },
          },
        ],
      },
      {
        type: "group",
        heading: "Chat (answer generation)",
        items: [
          {
            name: "Generation backend",
            desc:
              "How cited answers are generated. 'None' is fully offline (retrieval-only). " +
              "Ollama and LM Studio use a local server. Hosted is opt-in. Every non-None " +
              "backend sends only the retrieved chunks, never the whole vault.",
            control: {
              type: "dropdown",
              key: "generationBackend",
              options: Object.fromEntries(
                GENERATION_BACKENDS.map((id) => [id, GENERATION_BACKEND_LABELS[id]]),
              ),
            },
          },
          {
            name: "Ollama endpoint",
            desc: "Local Ollama server URL.",
            visible: backend("ollama"),
            control: { type: "text", key: "ollamaEndpoint" },
          },
          {
            name: "Ollama model",
            desc: "Pick from the models you have pulled (run 'ollama pull <model>' to add more).",
            visible: backend("ollama"),
            render: (setting) =>
              this.renderModelDropdown(setting, "ollamaModel", () =>
                listOllamaModels(this.settings.ollamaEndpoint, obsidianHttpClient),
              ),
          },
          {
            name: "LM Studio endpoint",
            desc: "LM Studio's local server base URL (Developer tab → Start Server).",
            visible: backend("lmstudio"),
            control: { type: "text", key: "lmstudioEndpoint" },
          },
          {
            name: "LM Studio model",
            desc: "Pick from the models currently loaded in LM Studio.",
            visible: backend("lmstudio"),
            render: (setting) =>
              this.renderModelDropdown(setting, "lmstudioModel", () =>
                listOpenAiModels(this.settings.lmstudioEndpoint, undefined, obsidianHttpClient),
              ),
          },
          {
            name: "Hosted endpoint",
            desc: "OpenAI-compatible chat completions URL. Only retrieved chunks are sent.",
            visible: backend("hosted"),
            control: { type: "text", key: "hostedEndpoint" },
          },
          {
            name: "Hosted model",
            desc: "Model name to request from the hosted endpoint.",
            visible: backend("hosted"),
            control: { type: "text", key: "hostedModel" },
          },
          {
            name: "Hosted API key",
            desc: "Stored locally in plugin settings. Sent only to the endpoint above.",
            visible: backend("hosted"),
            render: (setting) => this.renderSecret(setting, "hostedApiKey"),
          },
        ],
      },
      {
        type: "group",
        heading: "Advanced (tuning)",
        items: [
          {
            name: "Chunk size (tokens)",
            desc:
              `Approximate tokens per chunk (max ${MAX_CHUNK_TOKENS}). ` +
              "Takes effect on the next index rebuild.",
            control: { type: "number", key: "chunkTokens", min: 1, max: MAX_CHUNK_TOKENS, step: 1 },
          },
          {
            name: "Chunk overlap (tokens)",
            desc: "Token overlap between adjacent chunks. Takes effect on the next index rebuild.",
            control: {
              type: "number",
              key: "chunkOverlap",
              min: 0,
              max: MAX_CHUNK_TOKENS,
              step: 1,
            },
          },
          {
            name: "Hybrid alpha",
            desc: "Search blend: 1.0 is purely semantic, 0.0 is purely lexical (BM25).",
            control: { type: "slider", key: "hybridAlpha", min: 0, max: 1, step: 0.05 },
          },
          {
            name: "HNSW threshold",
            desc: "Chunk count above which the approximate HNSW index is used instead of exact cosine.",
            control: {
              type: "number",
              key: "hnswThreshold",
              min: 1,
              max: MAX_HNSW_THRESHOLD,
              step: 1,
            },
          },
        ],
      },
      {
        type: "group",
        heading: "Maintenance",
        items: [
          {
            name: "Index status",
            render: (setting) => {
              setting.setDesc(this.indexStatusText());
            },
          },
          {
            name: "Rebuild index",
            desc:
              "Notes are indexed automatically as you create, edit, rename, or delete them. " +
              "Use this to rebuild the whole index and embeddings from scratch — e.g. after " +
              "changing chunking or embedding settings, or if search results look stale. " +
              "You can close this window; indexing continues in the background (progress shows " +
              "in the status bar).",
            render: (setting) => this.renderRebuild(setting),
          },
        ],
      },
    ];
  }

  /** One-line summary of the current index size for the maintenance section. */
  private indexStatusText(): string {
    const { notes, chunks } = this.host.indexStats();
    if (notes === 0) {
      return "No notes indexed yet.";
    }
    const noteWord = notes === 1 ? "note" : "notes";
    const chunkWord = chunks === 1 ? "chunk" : "chunks";
    return `Indexed: ${notes} ${noteWord} · ${chunks} ${chunkWord}.`;
  }

  /** A masked text input bound to a secret setting, so keys are never shown in clear. */
  private renderSecret(setting: Setting, key: "hostedApiKey" | "remoteEmbeddingApiKey"): void {
    setting.addText((text) => {
      text.inputEl.type = "password";
      text.setValue(this.settings[key]).onChange((value) => void this.setControlValue(key, value));
    });
  }

  /**
   * A model picker backed by a dropdown that asynchronously lists the models a
   * server reports, plus a Refresh button to re-query it. The currently-saved
   * value is always selectable (even if the server is down), and if no model is
   * set yet the first listed model is chosen. Falls back to a clear hint when the
   * server is unreachable.
   */
  private renderModelDropdown(
    setting: Setting,
    key: "ollamaModel" | "lmstudioModel" | "remoteEmbeddingModel",
    fetcher: () => Promise<string[]>,
  ): void {
    // Re-queries the server and repopulates the dropdown; assigned when the
    // dropdown is built and reused by the Refresh button below.
    let load = async (): Promise<void> => {};

    setting.addDropdown((dropdown) => {
      const rebuild = (models: string[], placeholder?: string): void => {
        // Read the saved value fresh each time so a manual refresh preserves
        // the current selection even after the user changed it.
        const current = this.settings[key];
        dropdown.selectEl.empty();
        const list = [...new Set([current, ...models])].filter((m) => m.length > 0);
        if (list.length === 0) {
          dropdown.addOption("", placeholder ?? "No models found");
        } else {
          for (const model of list) {
            dropdown.addOption(model, model);
          }
        }
        const next = current.length > 0 && list.includes(current) ? current : (list[0] ?? "");
        dropdown.setValue(next);
        if (next !== current) {
          void this.setControlValue(key, next);
        }
      };

      dropdown.onChange((value) => void this.setControlValue(key, value));

      load = async (): Promise<void> => {
        rebuild([], "Loading models…");
        try {
          rebuild(await fetcher());
        } catch {
          rebuild([], "Server unreachable — is it running?");
        }
      };
      void load();
    });

    setting.addButton((button) =>
      button
        .setButtonText("Refresh")
        .setTooltip("Re-query the server for its current models")
        .onClick(() => void load()),
    );
  }

  /**
   * The remote model's vector dimension, with a Detect button that embeds a
   * probe to read it from the server. The dimension must match what the server
   * actually returns, so it is verifiable here rather than failing deep inside
   * an indexing run.
   */
  private renderDimension(setting: Setting): void {
    setting.addText((text) =>
      text
        .setPlaceholder("e.g. 768")
        .setValue(
          this.settings.remoteEmbeddingDim > 0 ? String(this.settings.remoteEmbeddingDim) : "",
        )
        .onChange((value) => {
          const parsed = Number.parseInt(value, 10);
          void this.setControlValue(
            "remoteEmbeddingDim",
            Number.isFinite(parsed) && parsed > 0 ? parsed : 0,
          );
        }),
    );
    setting.addButton((button) =>
      button
        .setButtonText("Detect")
        .setTooltip("Embed a short probe to read the model's dimensionality")
        .onClick(async () => {
          const s = this.settings;
          if (s.remoteEmbeddingModel.length === 0) {
            new Notice("Pick an embedding model first.");
            return;
          }
          button.setDisabled(true).setButtonText("Detecting…");
          try {
            const dim = await detectRemoteEmbeddingDim({
              protocol: s.remoteEmbeddingProtocol,
              endpoint: s.remoteEmbeddingEndpoint,
              model: s.remoteEmbeddingModel,
              apiKey: s.remoteEmbeddingApiKey,
              http: obsidianHttpClient,
            });
            await this.setControlValue("remoteEmbeddingDim", dim);
            new Notice(`Detected ${dim} dimensions. Rebuild the index to apply.`);
            // Re-render so the dimension field shows the detected value.
            this.update();
          } catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            new Notice(`Couldn't detect dimensions — ${detail}`);
          } finally {
            button.setDisabled(false).setButtonText("Detect");
          }
        }),
    );
  }

  /** The rebuild action; refreshes the status readout when it finishes. */
  private renderRebuild(setting: Setting): void {
    setting.addButton((button) => {
      button
        .setButtonText("Rebuild index")
        .setCta()
        .onClick(async () => {
          button.setDisabled(true).setButtonText("Rebuilding…");
          try {
            await this.host.requestReindex();
          } finally {
            button.setDisabled(false).setButtonText("Rebuild index");
            // Re-render so the Index status row reflects the new counts.
            this.update();
          }
        });
    });
  }
}
