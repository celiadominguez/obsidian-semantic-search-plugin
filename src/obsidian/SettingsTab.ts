/**
 * Settings tab exposing every configuration key. Each control reads from and
 * writes to the host plugin's settings and persists immediately.
 *
 * Organized by how often a setting is touched, so the everyday knobs come first
 * and the tuning ones stay out of the way:
 *   1. Search & indexing — embedding source/model, excluded folders
 *   2. Chat (answer generation) — backend + its fields
 *   3. Advanced (tuning) — chunking, hybrid blend, HNSW threshold
 *   4. Maintenance — index status + rebuild
 *
 * Changing the embedding model re-indexes automatically; chunking and other
 * tuning changes take effect on the next rebuild. Anything that sends data off
 * the machine (hosted generation, remote embeddings) is clearly labelled opt-in.
 */

import { Notice, normalizePath, type Plugin, PluginSettingTab, Setting } from "obsidian";
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
import type {
  EmbeddingModelId,
  EmbeddingSource,
  GenerationBackend,
  RemoteEmbeddingProtocol,
  VaultSleuthSettings,
} from "../core/types";

/** Human-readable labels for the chat backend dropdown (values stay the ids). */
const GENERATION_BACKEND_LABELS: Record<GenerationBackend, string> = {
  none: "None — offline, retrieval only (default)",
  ollama: "Ollama (local server)",
  lmstudio: "LM Studio (local server)",
  hosted: "Hosted API (opt-in, sends chunks out)",
};

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

  public display(): void {
    const { containerEl } = this;
    containerEl.empty();
    const settings = this.host.settings;

    // Ordered by how often it's touched: everyday indexing/search settings, then
    // chat, then power-user tuning, then maintenance actions.
    this.displaySearchIndexing(containerEl, settings);
    this.displayChat(containerEl, settings);
    this.displayAdvanced(containerEl, settings);
    this.displayMaintenance(containerEl);
  }

  /** Everyday settings: where embeddings come from, and what to index. */
  private displaySearchIndexing(containerEl: HTMLElement, settings: VaultSleuthSettings): void {
    new Setting(containerEl).setName("Search & indexing").setHeading();

    new Setting(containerEl)
      .setName("Embedding source")
      .setDesc(
        "Where embeddings are computed. 'On-device' runs the bundled model locally — " +
          "nothing leaves your machine. 'Remote server' delegates to an embeddings API: " +
          "private if you point it at localhost (Ollama / LM Studio), but it sends your " +
          "note text off your machine if you point it at a hosted service. " +
          "Switching clears the index — rebuild it afterwards.",
      )
      .addDropdown((dropdown) => {
        dropdown.addOption("on-device", "On-device (default)");
        dropdown.addOption("remote", "Remote server");
        dropdown.setValue(settings.embeddingSource).onChange(async (value) => {
          settings.embeddingSource = value as EmbeddingSource;
          await this.host.saveSettings();
          // Re-render so the source's own fields replace the other's.
          this.display();
        });
      });

    if (settings.embeddingSource === "on-device") {
      new Setting(containerEl)
        .setName("Embedding model")
        .setDesc("On-device model used to embed notes. Changing it triggers a full re-index.")
        .addDropdown((dropdown) => {
          for (const [id, info] of Object.entries(EMBEDDING_MODELS)) {
            dropdown.addOption(id, info.label);
          }
          dropdown.setValue(settings.embeddingModel).onChange(async (value) => {
            settings.embeddingModel = value as EmbeddingModelId;
            await this.host.saveSettings();
            await this.host.requestReindex();
          });
        });

      new Setting(containerEl)
        .setName("Local model folder (offline, advanced)")
        .setDesc(
          "Optional. A vault folder containing the model files, laid out as " +
            "<folder>/<model id>/… (e.g. onnx/model_quantized.onnx, config.json, " +
            "tokenizer.json). When set, the model loads from disk and is never " +
            "downloaded. Leave empty to download it once from Hugging Face. " +
            "Experimental — rebuild the index and confirm search still works after setting it.",
        )
        .addText((text) =>
          text
            .setPlaceholder("e.g. models")
            .setValue(settings.localModelPath)
            .onChange(async (value) => {
              const trimmed = value.trim();
              // Normalize user-entered paths; keep "" (disabled) as-is — normalizePath
              // turns an empty string into "/", which would look like a real folder.
              settings.localModelPath = trimmed.length > 0 ? normalizePath(trimmed) : "";
              await this.host.saveSettings();
            }),
        );
    } else {
      this.displayRemoteEmbedding(containerEl, settings);
    }

    new Setting(containerEl)
      .setName("Excluded folders")
      .setDesc(
        "Comma-separated vault paths to skip when indexing. Applies to new edits immediately; " +
          "rebuild the index (below) to drop already-indexed notes.",
      )
      .addTextArea((text) =>
        text.setValue(settings.excludedFolders.join(", ")).onChange(async (value) => {
          settings.excludedFolders = value
            .split(",")
            .map((folder) => folder.trim())
            .filter((folder) => folder.length > 0)
            .map((folder) => normalizePath(folder));
          await this.host.saveSettings();
        }),
      );
  }

  /** Chat backend selector plus the fields the chosen backend needs. */
  private displayChat(containerEl: HTMLElement, settings: VaultSleuthSettings): void {
    new Setting(containerEl).setName("Chat (answer generation)").setHeading();

    new Setting(containerEl)
      .setName("Generation backend")
      .setDesc(
        "How cited answers are generated. 'None' is fully offline (retrieval-only). " +
          "Ollama and LM Studio use a local server. Hosted is opt-in. Every non-None " +
          "backend sends only the retrieved chunks, never the whole vault.",
      )
      .addDropdown((dropdown) => {
        for (const backend of GENERATION_BACKENDS) {
          dropdown.addOption(backend, GENERATION_BACKEND_LABELS[backend]);
        }
        dropdown.setValue(settings.generationBackend).onChange(async (value) => {
          settings.generationBackend = value as GenerationBackend;
          await this.host.saveSettings();
          this.display();
        });
      });

    if (settings.generationBackend === "ollama") {
      new Setting(containerEl)
        .setName("Ollama endpoint")
        .setDesc("Local Ollama server URL.")
        .addText((text) =>
          text.setValue(settings.ollamaEndpoint).onChange(async (value) => {
            settings.ollamaEndpoint = value.trim();
            await this.host.saveSettings();
          }),
        );
      this.addModelDropdown(
        containerEl,
        "Ollama model",
        "Pick from the models you have pulled (run 'ollama pull <model>' to add more).",
        () => settings.ollamaModel,
        (value) => {
          settings.ollamaModel = value;
        },
        () => listOllamaModels(settings.ollamaEndpoint, obsidianHttpClient),
      );
    }

    if (settings.generationBackend === "lmstudio") {
      new Setting(containerEl)
        .setName("LM Studio endpoint")
        .setDesc("LM Studio's local server base URL (Developer tab → Start Server).")
        .addText((text) =>
          text.setValue(settings.lmstudioEndpoint).onChange(async (value) => {
            settings.lmstudioEndpoint = value.trim();
            await this.host.saveSettings();
          }),
        );
      this.addModelDropdown(
        containerEl,
        "LM Studio model",
        "Pick from the models currently loaded in LM Studio.",
        () => settings.lmstudioModel,
        (value) => {
          settings.lmstudioModel = value;
        },
        () => listOpenAiModels(settings.lmstudioEndpoint, undefined, obsidianHttpClient),
      );
    }

    if (settings.generationBackend === "hosted") {
      new Setting(containerEl)
        .setName("Hosted endpoint")
        .setDesc("OpenAI-compatible chat completions URL. Only retrieved chunks are sent.")
        .addText((text) =>
          text.setValue(settings.hostedEndpoint).onChange(async (value) => {
            settings.hostedEndpoint = value.trim();
            await this.host.saveSettings();
          }),
        );
      new Setting(containerEl)
        .setName("Hosted model")
        .setDesc("Model name to request from the hosted endpoint.")
        .addText((text) =>
          text.setValue(settings.hostedModel).onChange(async (value) => {
            settings.hostedModel = value.trim();
            await this.host.saveSettings();
          }),
        );
      new Setting(containerEl)
        .setName("Hosted API key")
        .setDesc("Stored locally in plugin settings. Sent only to the endpoint above.")
        .addText((text) => {
          text.inputEl.type = "password";
          text.setValue(settings.hostedApiKey).onChange(async (value) => {
            settings.hostedApiKey = value.trim();
            await this.host.saveSettings();
          });
        });
    }
  }

  /** Power-user tuning. Every field here has a sensible default; most users never touch it. */
  private displayAdvanced(containerEl: HTMLElement, settings: VaultSleuthSettings): void {
    new Setting(containerEl).setName("Advanced (tuning)").setHeading();

    new Setting(containerEl)
      .setName("Chunk size (tokens)")
      .setDesc(
        `Approximate tokens per chunk (max ${MAX_CHUNK_TOKENS}). ` +
          "Takes effect on the next index rebuild.",
      )
      .addText((text) =>
        text.setValue(String(settings.chunkTokens)).onChange(async (value) => {
          const parsed = Number.parseInt(value, 10);
          if (Number.isFinite(parsed) && parsed > 0) {
            const clamped = Math.min(parsed, MAX_CHUNK_TOKENS);
            settings.chunkTokens = clamped;
            if (clamped !== parsed) {
              text.setValue(String(clamped));
            }
            await this.host.saveSettings();
          }
        }),
      );

    new Setting(containerEl)
      .setName("Chunk overlap (tokens)")
      .setDesc("Token overlap between adjacent chunks. Takes effect on the next index rebuild.")
      .addText((text) =>
        text.setValue(String(settings.chunkOverlap)).onChange(async (value) => {
          const parsed = Number.parseInt(value, 10);
          if (Number.isFinite(parsed) && parsed >= 0) {
            const clamped = Math.min(parsed, MAX_CHUNK_TOKENS);
            settings.chunkOverlap = clamped;
            if (clamped !== parsed) {
              text.setValue(String(clamped));
            }
            await this.host.saveSettings();
          }
        }),
      );

    new Setting(containerEl)
      .setName("Hybrid alpha")
      .setDesc("Search blend: 1.0 is purely semantic, 0.0 is purely lexical (BM25).")
      .addSlider((slider) =>
        slider
          .setLimits(0, 1, 0.05)
          .setValue(settings.hybridAlpha)
          .setDynamicTooltip()
          .onChange(async (value) => {
            settings.hybridAlpha = value;
            await this.host.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("HNSW threshold")
      .setDesc(
        "Chunk count above which the approximate HNSW index is used instead of exact cosine.",
      )
      .addText((text) =>
        text.setValue(String(settings.hnswThreshold)).onChange(async (value) => {
          const parsed = Number.parseInt(value, 10);
          if (Number.isFinite(parsed) && parsed > 0) {
            const clamped = Math.min(parsed, MAX_HNSW_THRESHOLD);
            settings.hnswThreshold = clamped;
            if (clamped !== parsed) {
              text.setValue(String(clamped));
            }
            await this.host.saveSettings();
          }
        }),
      );
  }

  /** Index status readout and the rebuild action. */
  private displayMaintenance(containerEl: HTMLElement): void {
    new Setting(containerEl).setName("Maintenance").setHeading();

    const statusSetting = new Setting(containerEl)
      .setName("Index status")
      .setDesc(this.indexStatusText());

    new Setting(containerEl)
      .setName("Rebuild index")
      .setDesc(
        "Notes are indexed automatically as you create, edit, rename, or delete them. " +
          "Use this to rebuild the whole index and embeddings from scratch — e.g. after " +
          "changing chunking or embedding settings, or if search results look stale. " +
          "You can close this window; indexing continues in the background (progress shows " +
          "in the status bar).",
      )
      .addButton((button) => {
        button
          .setButtonText("Rebuild index")
          .setCta()
          .onClick(async () => {
            button.setDisabled(true).setButtonText("Rebuilding…");
            try {
              await this.host.requestReindex();
            } finally {
              button.setDisabled(false).setButtonText("Rebuild index");
              statusSetting.setDesc(this.indexStatusText());
            }
          });
      });
  }

  /**
   * Controls for the opt-in remote embeddings server: wire format, endpoint,
   * model, dimension (detectable), and an optional key. The dimension must match
   * what the server actually returns, so it is verifiable here rather than
   * failing deep inside an indexing run.
   */
  private displayRemoteEmbedding(containerEl: HTMLElement, settings: VaultSleuthSettings): void {
    new Setting(containerEl)
      .setName("Server type")
      .setDesc(
        "'OpenAI-compatible' works with LM Studio, Text Embeddings Inference, Infinity, " +
          "and hosted APIs (POST /embeddings). 'Ollama' uses its native POST /api/embed.",
      )
      .addDropdown((dropdown) => {
        dropdown.addOption("openai", "OpenAI-compatible");
        dropdown.addOption("ollama", "Ollama");
        dropdown.setValue(settings.remoteEmbeddingProtocol).onChange(async (value) => {
          settings.remoteEmbeddingProtocol = value as RemoteEmbeddingProtocol;
          await this.host.saveSettings();
          this.display();
        });
      });

    new Setting(containerEl)
      .setName("Embeddings endpoint")
      .setDesc(
        settings.remoteEmbeddingProtocol === "ollama"
          ? "Ollama base URL, e.g. http://localhost:11434"
          : "Base URL including the version prefix, e.g. http://localhost:1234/v1",
      )
      .addText((text) =>
        text.setValue(settings.remoteEmbeddingEndpoint).onChange(async (value) => {
          settings.remoteEmbeddingEndpoint = value.trim();
          await this.host.saveSettings();
        }),
      );

    this.addModelDropdown(
      containerEl,
      "Embedding model",
      "Pick an embedding model the server offers. Changing it clears the index — rebuild afterwards.",
      () => settings.remoteEmbeddingModel,
      (value) => {
        settings.remoteEmbeddingModel = value;
      },
      () =>
        settings.remoteEmbeddingProtocol === "ollama"
          ? listOllamaModels(settings.remoteEmbeddingEndpoint, obsidianHttpClient)
          : listOpenAiModels(
              settings.remoteEmbeddingEndpoint,
              settings.remoteEmbeddingApiKey,
              obsidianHttpClient,
            ),
    );

    new Setting(containerEl)
      .setName("Vector dimension")
      .setDesc(
        "Must match what the model returns. Use Detect to ask the server, then rebuild the index.",
      )
      .addText((text) =>
        text
          .setPlaceholder("e.g. 768")
          .setValue(settings.remoteEmbeddingDim > 0 ? String(settings.remoteEmbeddingDim) : "")
          .onChange(async (value) => {
            const parsed = Number.parseInt(value, 10);
            settings.remoteEmbeddingDim = Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
            await this.host.saveSettings();
          }),
      )
      .addButton((button) =>
        button
          .setButtonText("Detect")
          .setTooltip("Embed a short probe to read the model's dimensionality")
          .onClick(async () => {
            if (settings.remoteEmbeddingModel.length === 0) {
              new Notice("Pick an embedding model first.");
              return;
            }
            button.setDisabled(true).setButtonText("Detecting…");
            try {
              const dim = await detectRemoteEmbeddingDim({
                protocol: settings.remoteEmbeddingProtocol,
                endpoint: settings.remoteEmbeddingEndpoint,
                model: settings.remoteEmbeddingModel,
                apiKey: settings.remoteEmbeddingApiKey,
                http: obsidianHttpClient,
              });
              settings.remoteEmbeddingDim = dim;
              await this.host.saveSettings();
              new Notice(`Detected ${dim} dimensions. Rebuild the index to apply.`);
              this.display();
            } catch (error) {
              const detail = error instanceof Error ? error.message : String(error);
              new Notice(`Couldn't detect dimensions — ${detail}`);
            } finally {
              button.setDisabled(false).setButtonText("Detect");
            }
          }),
      );

    new Setting(containerEl)
      .setName("API key (optional)")
      .setDesc(
        "Only needed for servers that require auth. Stored locally and sent " +
          "as a bearer token to the endpoint above, nowhere else.",
      )
      .addText((text) => {
        text.inputEl.type = "password";
        text.setValue(settings.remoteEmbeddingApiKey).onChange(async (value) => {
          settings.remoteEmbeddingApiKey = value.trim();
          await this.host.saveSettings();
        });
      });

    new Setting(containerEl)
      .setName("Query instruction (advanced)")
      .setDesc(
        "Prepended to your query (not your notes) before embedding, for asymmetric " +
          "models like the BGE family — e.g. 'Represent this sentence for searching " +
          "relevant passages: '. Leave empty for symmetric models. Applies to queries " +
          "only, so no re-index is needed.",
      )
      .addText((text) =>
        text
          .setPlaceholder("Represent this sentence for searching relevant passages: ")
          .setValue(settings.remoteEmbeddingQueryInstruction)
          .onChange(async (value) => {
            settings.remoteEmbeddingQueryInstruction = value;
            await this.host.saveSettings();
          }),
      );
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

  /**
   * A model setting backed by a dropdown that asynchronously lists the models a
   * local server reports. The currently-saved value is always selectable (even if
   * the server is down), and if no model is set yet the first listed model is
   * chosen. Falls back to a clear hint when the server is unreachable.
   */
  private addModelDropdown(
    containerEl: HTMLElement,
    name: string,
    desc: string,
    getValue: () => string,
    setValue: (value: string) => void,
    fetcher: () => Promise<string[]>,
  ): void {
    // Re-queries the server and repopulates the dropdown; assigned when the
    // dropdown is built and reused by the Refresh button below.
    let load = async (): Promise<void> => {};

    const setting = new Setting(containerEl)
      .setName(name)
      .setDesc(desc)
      .addDropdown((dropdown) => {
        const rebuild = (models: string[], placeholder?: string): void => {
          // Read the saved value fresh each time so a manual refresh preserves
          // the current selection even after the user changed it.
          const current = getValue();
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
            setValue(next);
            void this.host.saveSettings();
          }
        };

        dropdown.onChange(async (value) => {
          setValue(value);
          await this.host.saveSettings();
        });

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
}
