/**
 * The VaultSeek sidebar view: a search box, ranked results with previews and
 * per-result actions (open in split, insert as link, copy citation), and a
 * cited Q&A panel. All retrieval and Q&A work is delegated to `IndexService`;
 * this file is thin presentation glue over Obsidian's `ItemView`.
 */

import { ItemView, MarkdownRenderer, Notice, TFile, type WorkspaceLeaf } from "obsidian";
import { DEFAULT_TOP_K } from "../core/config";
import { noteBasename } from "../core/notePath";
import type { QaResult, SearchResult } from "../core/types";
import type { IndexService } from "./indexService";

/** Stable view type id used to register and reveal the view. */
export const VAULTSEEK_VIEW_TYPE = "vaultseek-search-view";

export class SearchView extends ItemView {
  private readonly index: IndexService;
  private searchInput!: HTMLInputElement;
  private statusEl!: HTMLElement;
  private resultsEl!: HTMLElement;
  private qaEl!: HTMLElement;
  private debounceTimer: number | null = null;

  constructor(leaf: WorkspaceLeaf, index: IndexService) {
    super(leaf);
    this.index = index;
  }

  public getViewType(): string {
    return VAULTSEEK_VIEW_TYPE;
  }

  public getDisplayText(): string {
    return "VaultSeek";
  }

  public getIcon(): string {
    return "search";
  }

  public async onOpen(): Promise<void> {
    const container = this.contentEl;
    container.empty();
    container.addClass("vaultseek-view");

    const searchRow = container.createDiv({ cls: "vaultseek-search-row" });
    this.searchInput = searchRow.createEl("input", {
      cls: "vaultseek-search-input",
      attr: { type: "text", placeholder: "Search your vault by meaning…" },
    });
    const askButton = searchRow.createEl("button", { text: "Ask" });

    this.statusEl = container.createDiv({ cls: "vaultseek-status" });
    this.resultsEl = container.createDiv({ cls: "vaultseek-results" });
    this.qaEl = container.createDiv({ cls: "vaultseek-qa" });

    this.searchInput.addEventListener("input", () => this.scheduleSearch());
    this.searchInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        void this.runQuestion();
      }
    });
    askButton.addEventListener("click", () => void this.runQuestion());

    this.renderEmptyState();
  }

  public async onClose(): Promise<void> {
    if (this.debounceTimer !== null) {
      window.clearTimeout(this.debounceTimer);
    }
  }

  /** Focus the search box and optionally seed it (used by the command). */
  public focusSearch(initialQuery?: string): void {
    if (initialQuery !== undefined) {
      this.searchInput.value = initialQuery;
      void this.runSearch();
    }
    this.searchInput.focus();
  }

  private scheduleSearch(): void {
    if (this.debounceTimer !== null) {
      window.clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = window.setTimeout(() => void this.runSearch(), 200);
  }

  private renderEmptyState(): void {
    this.resultsEl.empty();
    this.resultsEl.createDiv({
      cls: "vaultseek-empty",
      text: "Type to search semantically, or press Enter / Ask for a cited answer.",
    });
  }

  private async runSearch(): Promise<void> {
    const query = this.searchInput.value.trim();
    this.qaEl.empty();
    if (query.length === 0) {
      this.renderEmptyState();
      this.statusEl.setText("");
      return;
    }
    this.statusEl.setText("Searching…");
    const results = await this.index.search(query, "hybrid", DEFAULT_TOP_K);
    this.statusEl.setText(`${results.length} result${results.length === 1 ? "" : "s"}`);
    this.renderResults(results);
  }

  private renderResults(results: SearchResult[]): void {
    this.resultsEl.empty();
    if (results.length === 0) {
      this.resultsEl.createDiv({ cls: "vaultseek-empty", text: "No matches found." });
      return;
    }
    for (const result of results) {
      this.renderResult(result);
    }
  }

  private renderResult(result: SearchResult): void {
    const card = this.resultsEl.createDiv({ cls: "vaultseek-result" });
    const header = card.createDiv({ cls: "vaultseek-result-header" });
    header.createDiv({ cls: "vaultseek-result-title", text: result.chunk.noteTitle });
    header.createDiv({ cls: "vaultseek-result-score", text: result.score.toFixed(3) });
    if (result.chunk.heading.length > 0) {
      card.createDiv({ cls: "vaultseek-result-snippet", text: `# ${result.chunk.heading}` });
    }
    card.createDiv({ cls: "vaultseek-result-snippet", text: result.snippet });

    const actions = card.createDiv({ cls: "vaultseek-result-actions" });
    this.addResultAction(actions, "Open in split", () => this.openInSplit(result.chunk.notePath));
    this.addResultAction(actions, "Insert link", () => this.insertLink(result.chunk.notePath));
    this.addResultAction(actions, "Copy citation", () => this.copyCitation(result.chunk.notePath));

    card.addEventListener("click", (event) => {
      if (!(event.target as HTMLElement).hasClass("vaultseek-result-action")) {
        void this.openInSplit(result.chunk.notePath);
      }
    });
  }

  private addResultAction(parent: HTMLElement, label: string, handler: () => void): void {
    const el = parent.createSpan({ cls: "vaultseek-result-action", text: label });
    el.addEventListener("click", (event) => {
      event.stopPropagation();
      handler();
    });
  }

  private async runQuestion(): Promise<void> {
    const question = this.searchInput.value.trim();
    if (question.length === 0) {
      return;
    }
    await this.runSearch();
    this.qaEl.empty();
    this.qaEl.createEl("h4", { text: "Answer" });
    const answerEl = this.qaEl.createDiv({ cls: "vaultseek-qa-answer" });
    answerEl.setText("Thinking…");

    let result: QaResult;
    try {
      result = await this.index.answer(question);
    } catch (error) {
      answerEl.setText(error instanceof Error ? error.message : String(error));
      return;
    }

    answerEl.empty();
    answerEl.toggleClass("is-refusal", result.refused);
    await MarkdownRenderer.render(this.app, result.answer, answerEl, "", this);

    if (result.citations.length > 0) {
      const cites = this.qaEl.createDiv({ cls: "vaultseek-qa-citations" });
      cites.setText(`Sources: ${result.citations.map((path) => noteBasename(path)).join(", ")}`);
    }
  }

  private getFile(path: string): TFile | null {
    const file = this.app.vault.getAbstractFileByPath(path);
    return file instanceof TFile ? file : null;
  }

  private async openInSplit(path: string): Promise<void> {
    const file = this.getFile(path);
    if (file === null) {
      new Notice(`VaultSeek: note not found — ${path}`);
      return;
    }
    await this.app.workspace.getLeaf("split").openFile(file);
  }

  private insertLink(path: string): void {
    const markdownView = this.app.workspace.activeEditor;
    const link = `[[${noteBasename(path)}]]`;
    if (markdownView?.editor) {
      markdownView.editor.replaceSelection(link);
      new Notice("VaultSeek: link inserted");
    } else {
      new Notice("VaultSeek: open a note to insert a link");
    }
  }

  private async copyCitation(path: string): Promise<void> {
    await navigator.clipboard.writeText(`[[${noteBasename(path)}]]`);
    new Notice("VaultSeek: citation copied");
  }
}
