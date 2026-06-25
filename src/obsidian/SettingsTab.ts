/**
 * Settings tab exposing every configuration key. Each control reads from and
 * writes to the host plugin's settings and persists immediately. Changing the
 * embedding model re-indexes automatically; chunking changes take effect on the
 * next manual re-index. Privacy-sensitive controls (hosted generation) are
 * grouped and clearly labelled as opt-in.
 */

import { type Plugin, PluginSettingTab, Setting } from "obsidian";
import {
  EMBEDDING_MODELS,
  GENERATION_BACKENDS,
  MAX_CHUNK_TOKENS,
  MAX_HNSW_THRESHOLD,
} from "../core/config";
import { listOllamaModels, listOpenAiModels } from "../core/generation";
import { obsidianHttpClient } from "./obsidianHttp";
import type { EmbeddingModelId, GenerationBackend, VaultSleuthSettings } from "../core/types";

/** What the settings tab needs from the plugin, decoupled from the class. */
export interface SettingsHost extends Plugin {
  settings: VaultSleuthSettings;
  saveSettings(): Promise<void>;
  /** Re-embed the whole vault (after a model or chunking change). */
  requestReindex(): Promise<void>;
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

    new Setting(containerEl).setName("Embedding").setHeading();

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
          "Experimental — re-index and confirm search still works after setting it.",
      )
      .addText((text) =>
        text
          .setPlaceholder("e.g. models")
          .setValue(settings.localModelPath)
          .onChange(async (value) => {
            settings.localModelPath = value.trim();
            await this.host.saveSettings();
          }),
      );

    new Setting(containerEl).setName("Chunking").setHeading();

    new Setting(containerEl)
      .setName("Chunk size (tokens)")
      .setDesc(
        `Approximate tokens per chunk (max ${MAX_CHUNK_TOKENS}). ` +
          'Takes effect on the next "Re-index vault".',
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
      .setDesc('Token overlap between adjacent chunks. Takes effect on the next "Re-index vault".')
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

    new Setting(containerEl).setName("Ranking").setHeading();

    new Setting(containerEl)
      .setName("Hybrid alpha")
      .setDesc("Blend weight: 1.0 is purely semantic, 0.0 is purely lexical (BM25).")
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

    new Setting(containerEl)
      .setName("Excluded folders")
      .setDesc(
        "Comma-separated vault paths to skip when indexing. Applies to new edits immediately; " +
          'run "Re-index vault" to drop already-indexed notes.',
      )
      .addTextArea((text) =>
        text.setValue(settings.excludedFolders.join(", ")).onChange(async (value) => {
          settings.excludedFolders = value
            .split(",")
            .map((folder) => folder.trim())
            .filter((folder) => folder.length > 0);
          await this.host.saveSettings();
        }),
      );

    new Setting(containerEl).setName("Chat (answer generation)").setHeading();

    new Setting(containerEl)
      .setName("Generation backend")
      .setDesc(
        "How cited answers are generated. 'none' is fully offline (retrieval-only). " +
          "'ollama' and 'lmstudio' use a local server. 'hosted' is opt-in. All non-'none' " +
          "backends send only the retrieved chunks, never the whole vault.",
      )
      .addDropdown((dropdown) => {
        for (const backend of GENERATION_BACKENDS) {
          dropdown.addOption(backend, backend);
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
    new Setting(containerEl)
      .setName(name)
      .setDesc(desc)
      .addDropdown((dropdown) => {
        const current = getValue();

        const rebuild = (models: string[], placeholder?: string): void => {
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

        rebuild([], "Loading models…");
        dropdown.onChange(async (value) => {
          setValue(value);
          await this.host.saveSettings();
        });
        void fetcher()
          .then((models) => rebuild(models))
          .catch(() => rebuild([], "Server unreachable — is it running?"));
      });
  }
}
