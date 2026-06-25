/**
 * VaultSleuth plugin entry point.
 *
 * Wires the Obsidian runtime to the indexing service and views: registers the
 * search view, a command-palette entry and ribbon icon, the settings tab, and
 * debounced vault change handlers that keep the index incrementally up to date.
 * It also owns settings persistence and the index lifecycle (load on start,
 * re-index on demand). Views, commands, and vault events are registered via the
 * plugin's `register*` helpers, so Obsidian tears them down automatically.
 */

import { Notice, Plugin, TFile, type WorkspaceLeaf } from "obsidian";
import { INDEX_DEBOUNCE_MS } from "./core/config";
import { defaultSettings } from "./core/config";
import type { VaultSleuthSettings } from "./core/types";
import { IndexService } from "./obsidian/indexService";
import { VaultSleuthView, VAULTSLEUTH_VIEW_TYPE, type ViewMode } from "./obsidian/VaultSleuthView";
import { SettingsTab, type SettingsHost } from "./obsidian/SettingsTab";

const DEFAULT_PLUGIN_DIR = ".obsidian/plugins/vaultsleuth";

export default class VaultSleuthPlugin extends Plugin implements SettingsHost {
  public settings: VaultSleuthSettings = defaultSettings();
  private index!: IndexService;
  private statusBar!: HTMLElement;
  private readonly pendingPaths = new Set<string>();
  private debounceTimer: number | null = null;

  public async onload(): Promise<void> {
    await this.loadSettings();

    const pluginDir = this.manifest.dir ?? DEFAULT_PLUGIN_DIR;
    this.index = new IndexService(this.app, this.settings, pluginDir);

    this.registerView(VAULTSLEUTH_VIEW_TYPE, (leaf) => new VaultSleuthView(leaf, this.index));

    this.addRibbonIcon(
      "brain-circuit",
      "VaultSleuth: search & chat",
      () => void this.activateView("search"),
    );

    this.addCommand({
      id: "open-search",
      name: "Open semantic search",
      callback: () => void this.activateView("search"),
    });

    this.addCommand({
      id: "open-chat",
      name: "Open chat",
      callback: () => void this.activateView("chat"),
    });

    this.addCommand({
      id: "reindex-vault",
      name: "Re-index vault",
      callback: () => void this.requestReindex(),
    });

    this.addSettingTab(new SettingsTab(this));

    this.statusBar = this.addStatusBarItem();

    this.registerVaultEvents();

    // Defer initial indexing until the workspace is ready so startup is snappy.
    this.app.workspace.onLayoutReady(() => void this.bootstrapIndex());
  }

  private async bootstrapIndex(): Promise<void> {
    const loaded = await this.index.loadPersisted();
    if (!loaded) {
      await this.requestReindex();
    } else {
      this.setStatus(`indexed (${this.index.stats().chunks} chunks)`);
    }
  }

  /** Open (or reveal) the unified view in the right sidebar, set to `mode`. */
  private async activateView(mode: ViewMode): Promise<void> {
    const { workspace } = this.app;
    let leaf: WorkspaceLeaf | null = workspace.getLeavesOfType(VAULTSLEUTH_VIEW_TYPE)[0] ?? null;
    if (leaf === null) {
      leaf = workspace.getRightLeaf(false);
      await leaf?.setViewState({ type: VAULTSLEUTH_VIEW_TYPE, active: true });
    }
    if (leaf !== null) {
      await workspace.revealLeaf(leaf);
      if (leaf.view instanceof VaultSleuthView) {
        leaf.view.setMode(mode);
      }
    }
  }

  private registerVaultEvents(): void {
    this.registerEvent(
      this.app.vault.on("modify", (file) => {
        if (file instanceof TFile && file.extension === "md") {
          this.queue(file.path);
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("create", (file) => {
        if (file instanceof TFile && file.extension === "md") {
          this.queue(file.path);
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("delete", (file) => {
        if (file instanceof TFile && file.extension === "md") {
          this.index.removeNote(file.path);
          void this.index.persist();
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => {
        if (file instanceof TFile && file.extension === "md") {
          void this.index.renameNote(oldPath, file).then(() => this.index.persist());
        }
      }),
    );
  }

  private queue(path: string): void {
    this.pendingPaths.add(path);
    if (this.debounceTimer !== null) {
      window.clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = window.setTimeout(() => void this.flushQueue(), INDEX_DEBOUNCE_MS);
  }

  private async flushQueue(): Promise<void> {
    const paths = [...this.pendingPaths];
    this.pendingPaths.clear();
    for (const path of paths) {
      const file = this.app.vault.getAbstractFileByPath(path);
      if (file instanceof TFile) {
        await this.index.indexFile(file);
      }
    }
    await this.index.persist();
    this.setStatus(`indexed (${this.index.stats().chunks} chunks)`);
  }

  /** Re-embed the entire vault, surfacing progress in the status bar. */
  public async requestReindex(): Promise<void> {
    new Notice("VaultSleuth: indexing vault…");
    await this.index.reindexAll((done, total) => {
      this.setStatus(`indexing ${done}/${total}`);
    });
    const { chunks, notes } = this.index.stats();
    this.setStatus(`indexed (${chunks} chunks)`);
    new Notice(`VaultSleuth: indexed ${notes} notes (${chunks} chunks)`);
  }

  private setStatus(text: string): void {
    this.statusBar.setText(`VaultSleuth: ${text}`);
  }

  public async loadSettings(): Promise<void> {
    this.settings = Object.assign(defaultSettings(), await this.loadData());
  }

  public async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
    this.index?.updateSettings(this.settings);
    // Reflect backend changes (e.g. enabling/disabling chat) in any open view.
    for (const leaf of this.app.workspace.getLeavesOfType(VAULTSLEUTH_VIEW_TYPE)) {
      if (leaf.view instanceof VaultSleuthView) {
        leaf.view.updateChatChrome();
      }
    }
  }
}
