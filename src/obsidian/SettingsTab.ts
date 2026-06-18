/**
 * Settings tab exposing every configuration key. Each control reads from and
 * writes to the host plugin's settings and persists immediately. Changes that
 * affect the index (model, chunking) prompt a re-index; privacy-sensitive
 * controls (hosted generation) are grouped and clearly labelled as opt-in.
 */

import { type Plugin, PluginSettingTab, Setting } from "obsidian";
import { EMBEDDING_MODELS, GENERATION_BACKENDS } from "../core/config";
import type { EmbeddingModelId, GenerationBackend, VaultSeekSettings } from "../core/types";

/** What the settings tab needs from the plugin, decoupled from the class. */
export interface SettingsHost extends Plugin {
  settings: VaultSeekSettings;
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
      .setName("Use WebGPU")
      .setDesc("Accelerate embedding with WebGPU when available. Falls back to WASM automatically.")
      .addToggle((toggle) =>
        toggle.setValue(settings.useWebGPU).onChange(async (value) => {
          settings.useWebGPU = value;
          await this.host.saveSettings();
        }),
      );

    new Setting(containerEl).setName("Chunking").setHeading();

    new Setting(containerEl)
      .setName("Chunk size (tokens)")
      .setDesc("Approximate tokens per chunk. Changing it triggers a full re-index.")
      .addText((text) =>
        text.setValue(String(settings.chunkTokens)).onChange(async (value) => {
          const parsed = Number.parseInt(value, 10);
          if (Number.isFinite(parsed) && parsed > 0) {
            settings.chunkTokens = parsed;
            await this.host.saveSettings();
          }
        }),
      );

    new Setting(containerEl)
      .setName("Chunk overlap (tokens)")
      .setDesc("Token overlap between adjacent chunks of one section.")
      .addText((text) =>
        text.setValue(String(settings.chunkOverlap)).onChange(async (value) => {
          const parsed = Number.parseInt(value, 10);
          if (Number.isFinite(parsed) && parsed >= 0) {
            settings.chunkOverlap = parsed;
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
            settings.hnswThreshold = parsed;
            await this.host.saveSettings();
          }
        }),
      );

    new Setting(containerEl)
      .setName("Excluded folders")
      .setDesc("Comma-separated vault paths to skip when indexing.")
      .addTextArea((text) =>
        text.setValue(settings.excludedFolders.join(", ")).onChange(async (value) => {
          settings.excludedFolders = value
            .split(",")
            .map((folder) => folder.trim())
            .filter((folder) => folder.length > 0);
          await this.host.saveSettings();
        }),
      );

    new Setting(containerEl).setName("Q&A generation").setHeading();

    new Setting(containerEl)
      .setName("Generation backend")
      .setDesc(
        "How cited answers are generated. 'none' is fully offline (retrieval-only). " +
          "'ollama' uses a local server. 'hosted' is opt-in and sends only retrieved chunks.",
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
      new Setting(containerEl)
        .setName("Ollama model")
        .setDesc("Model name served by Ollama, e.g. llama3.1:8b.")
        .addText((text) =>
          text.setValue(settings.ollamaModel).onChange(async (value) => {
            settings.ollamaModel = value.trim();
            await this.host.saveSettings();
          }),
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
}
