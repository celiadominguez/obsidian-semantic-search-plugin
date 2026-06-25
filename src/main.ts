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
import { INDEX_DEBOUNCE_MS, defaultSettings } from "./core/config";
import type { VaultSleuthSettings } from "./core/types";
import { IndexService } from "./obsidian/indexService";
import { VaultSleuthView, VAULTSLEUTH_VIEW_TYPE, type ViewMode } from "./obsidian/VaultSleuthView";
import { SettingsTab, type SettingsHost } from "./obsidian/SettingsTab";

export default class VaultSleuthPlugin extends Plugin implements SettingsHost {
  public settings: VaultSleuthSettings = defaultSettings();
  private index!: IndexService;
  private statusBar!: HTMLElement;
  private readonly pendingPaths = new Set<string>();
  private debounceTimer: number | null = null;

  public async onload(): Promise<void> {
    await this.loadSettings();

    // Obsidian sets manifest.dir for a loaded plugin; the fallback derives the
    // path from the (user-configurable) config dir and the manifest id rather
    // than hardcoding ".obsidian/plugins/vaultsleuth".
    const pluginDir =
      this.manifest.dir ?? `${this.app.vault.configDir}/plugins/${this.manifest.id}`;
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

  public onunload(): void {
    // Cancel any pending debounced re-index so it can't fire after teardown.
    if (this.debounceTimer !== null) {
      window.clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }

  private async bootstrapIndex(): Promise<void> {
    const loaded = await this.index.loadPersisted();
    if (loaded) {
      this.setStatus(`indexed (${this.index.stats().chunks} chunks)`);
      return;
    }
    // First run (or a reset index): the embedding model is fetched once (~33 MB)
    // from the Hugging Face CDN and cached on disk, then the vault is embedded
    // on-device. Announce it up front so the one-time download isn't a surprise.
    new Notice(
      "VaultSleuth: first-time setup — fetching the embedding model (~33 MB, once) if needed, " +
        "then indexing your vault on-device. This runs in the background.",
      10000,
    );
    await this.requestReindex(false);
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
          void this.index
            .removeNote(file.path)
            .catch((error) => this.reportIndexError(error))
            .finally(() => this.notifyIndexingState());
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => {
        if (file instanceof TFile && file.extension === "md") {
          void this.index
            .renameNote(oldPath, file)
            .catch((error) => this.reportIndexError(error))
            .finally(() => this.notifyIndexingState());
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
    const files: TFile[] = [];
    for (const path of paths) {
      const file = this.app.vault.getAbstractFileByPath(path);
      if (file instanceof TFile) {
        files.push(file);
      }
    }
    const indexing = this.index.indexFiles(files);
    this.notifyIndexingState();
    try {
      await indexing;
      this.setStatus(`indexed (${this.index.stats().chunks} chunks)`);
    } catch (error) {
      this.reportIndexError(error);
    } finally {
      this.notifyIndexingState();
    }
  }

  /**
   * Re-embed the entire vault, surfacing progress in the status bar.
   * `announceStart` shows the generic "Indexing vault…" notice; the first-run
   * bootstrap sets it false because it shows its own (download-aware) notice.
   */
  public async requestReindex(announceStart = true): Promise<void> {
    if (announceStart) {
      new Notice("Indexing vault…");
    }
    const indexing = this.index.reindexAll((done, total) => {
      this.setStatus(`indexing ${done}/${total}`);
    });
    this.notifyIndexingState();
    try {
      await indexing;
    } catch (error) {
      this.reportIndexError(error);
      new Notice(`VaultSleuth: indexing failed — ${this.errorMessage(error)}`);
      return;
    } finally {
      this.notifyIndexingState();
    }
    const { chunks, notes } = this.index.stats();
    if (notes === 0) {
      this.setStatus("no notes to index");
      return;
    }
    this.setStatus(`indexed (${chunks} chunks)`);
    new Notice(`Indexed ${notes} notes (${chunks} chunks)`);
  }

  private setStatus(text: string): void {
    this.statusBar.setText(`VaultSleuth: ${text}`);
  }

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  /**
   * Surface a background-indexing failure (e.g. the one-time embedding-model
   * download failing) in the status bar instead of leaving it stuck mid-progress
   * with an unhandled rejection.
   */
  private reportIndexError(error: unknown): void {
    console.error("VaultSleuth: indexing failed", error);
    this.setStatus(`indexing failed — ${this.errorMessage(error)}`);
  }

  /** Tell open views that indexing started/finished so they can warn and refresh. */
  private notifyIndexingState(): void {
    for (const leaf of this.app.workspace.getLeavesOfType(VAULTSLEUTH_VIEW_TYPE)) {
      if (leaf.view instanceof VaultSleuthView) {
        leaf.view.onIndexingStateChanged();
      }
    }
  }

  public async loadSettings(): Promise<void> {
    this.settings = Object.assign(defaultSettings(), await this.loadData());
  }

  public async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
    this.index?.updateSettings(this.settings);
    // Reflect backend/model changes in any open view: rebuild its chat engine
    // against the new settings and enable/disable the Chat tab accordingly.
    for (const leaf of this.app.workspace.getLeavesOfType(VAULTSLEUTH_VIEW_TYPE)) {
      if (leaf.view instanceof VaultSleuthView) {
        leaf.view.onSettingsChanged();
      }
    }
  }
}
