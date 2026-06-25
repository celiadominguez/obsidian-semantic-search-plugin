/**
 * The unified VaultSleuth panel: a single sidebar view with a Search ⇄ Chat toggle
 * sharing one input box.
 *
 *  - **Search** mode is pure find-and-navigate: ranked results with scores,
 *    snippets, and actions (open in split, insert link, copy citation).
 *  - **Chat** mode is a multi-turn, vault-grounded conversation with `[[note]]`
 *    citations and weak-retrieval refusal.
 *
 * Both modes share the same index and embedder via `IndexService`; this file is
 * thin presentation glue over Obsidian's `ItemView`.
 */

import { ItemView, MarkdownRenderer, Notice, TFile, type WorkspaceLeaf } from "obsidian";
import { DEFAULT_TOP_K } from "../core/config";
import { noteBasename } from "../core/notePath";
import type { ChatEngine } from "../core/chat";
import type { ChatMessage, SearchResult } from "../core/types";
import type { IndexService } from "./indexService";

/** Stable view type id used to register and reveal the view. */
export const VAULTSLEUTH_VIEW_TYPE = "vaultsleuth-view";

/** Which mode the shared panel is showing. */
export type ViewMode = "search" | "chat";

export class VaultSleuthView extends ItemView {
  private readonly index: IndexService;
  private mode: ViewMode = "search";
  private chatEngine: ChatEngine | null = null;
  /** Per-mode input drafts, so a search query never carries over into chat. */
  private readonly drafts: Record<ViewMode, string> = { search: "", chat: "" };

  private searchTab!: HTMLElement;
  private chatTab!: HTMLElement;
  private inputEl!: HTMLTextAreaElement;
  private primaryButton!: HTMLButtonElement;
  private resetButton!: HTMLButtonElement;
  private statusEl!: HTMLElement;
  private modelEl!: HTMLElement;
  private indexingEl!: HTMLElement;
  private searchBodyEl!: HTMLElement;
  private chatBodyEl!: HTMLElement;

  private debounceTimer: number | null = null;
  /** Monotonic id so a slow search can't overwrite the results of a newer one. */
  private searchSeq = 0;
  private sending = false;

  constructor(leaf: WorkspaceLeaf, index: IndexService) {
    super(leaf);
    this.index = index;
  }

  public getViewType(): string {
    return VAULTSLEUTH_VIEW_TYPE;
  }

  public getDisplayText(): string {
    return "VaultSleuth";
  }

  public getIcon(): string {
    return "brain-circuit";
  }

  public async onOpen(): Promise<void> {
    const container = this.contentEl;
    container.empty();
    container.addClass("vaultsleuth-view");

    const header = container.createDiv({ cls: "vaultsleuth-header" });
    const tabs = header.createDiv({ cls: "vaultsleuth-tabs" });
    this.searchTab = tabs.createEl("button", { text: "Search", cls: "vaultsleuth-tab" });
    this.chatTab = tabs.createEl("button", { text: "Chat", cls: "vaultsleuth-tab" });
    this.resetButton = header.createEl("button", { text: "New chat", cls: "vaultsleuth-reset" });

    this.searchTab.addEventListener("click", () => this.setMode("search"));
    this.chatTab.addEventListener("click", () => {
      if (this.chatAvailable()) {
        this.setMode("chat");
      }
    });
    this.resetButton.addEventListener("click", () => this.resetConversation());

    const inputRow = container.createDiv({ cls: "vaultsleuth-search-row" });
    this.inputEl = inputRow.createEl("textarea", {
      cls: "vaultsleuth-input",
      attr: { rows: "1", placeholder: "Search your vault by meaning…" },
    });
    this.primaryButton = inputRow.createEl("button", { text: "Search", cls: "mod-cta" });

    this.modelEl = container.createDiv({ cls: "vaultsleuth-chat-model" });
    this.statusEl = container.createDiv({ cls: "vaultsleuth-status" });
    // Warns (in both modes) that results may be incomplete while indexing runs.
    this.indexingEl = container.createDiv({ cls: "vaultsleuth-indexing-notice" });
    this.indexingEl.hide();
    // Separate, persistent containers so switching modes never loses either side.
    this.searchBodyEl = container.createDiv({ cls: "vaultsleuth-body" });
    this.chatBodyEl = container.createDiv({ cls: "vaultsleuth-body" });

    this.inputEl.addEventListener("input", () => {
      if (this.mode === "search") {
        this.scheduleSearch();
      }
    });
    this.inputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        void this.submit();
      }
    });
    this.primaryButton.addEventListener("click", () => void this.submit());

    this.setMode(this.mode);
  }

  public async onClose(): Promise<void> {
    if (this.debounceTimer !== null) {
      window.clearTimeout(this.debounceTimer);
    }
  }

  /** Whether chat is usable — only when a generative backend (not `none`) is set. */
  private chatAvailable(): boolean {
    return this.index.hasGenerativeBackend;
  }

  /** Switch the active mode, swapping in that mode's own input draft. */
  public setMode(mode: ViewMode): void {
    // Chat needs a model; without one, fall back to search.
    if (mode === "chat" && !this.chatAvailable()) {
      mode = "search";
    }
    // Cancel any pending search so it can't fire against the other mode's input.
    if (this.debounceTimer !== null) {
      window.clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    // Stash the outgoing mode's draft and restore the incoming mode's.
    this.drafts[this.mode] = this.inputEl.value;
    this.mode = mode;
    this.inputEl.value = this.drafts[mode];
    this.searchTab.toggleClass("is-active", mode === "search");
    this.chatTab.toggleClass("is-active", mode === "chat");
    this.resetButton.toggle(mode === "chat");
    this.primaryButton.setText(mode === "chat" ? "Send" : "Search");
    this.inputEl.setAttribute(
      "placeholder",
      mode === "chat" ? "Ask a question about your notes…" : "Search your vault by meaning…",
    );
    this.statusEl.setText("");
    this.updateChatChrome();
    this.refreshIndexingNotice();
    // Toggle visibility — both bodies persist, so neither side is ever lost.
    this.searchBodyEl.toggle(mode === "search");
    this.chatBodyEl.toggle(mode === "chat");
    if (mode === "search" && this.searchBodyEl.childElementCount === 0) {
      this.renderSearchEmpty();
    }
    if (mode === "chat" && this.chatBodyEl.childElementCount === 0) {
      this.renderChatEmpty();
    }
    this.inputEl.focus();
  }

  /**
   * Settings changed: drop the cached chat engine so the next turn is rebuilt
   * against the current model/backend/index, then refresh the chrome. Separate
   * from {@link updateChatChrome} (which runs on every mode switch) so toggling
   * Search/Chat never discards an in-progress conversation.
   */
  public onSettingsChanged(): void {
    this.chatEngine = null;
    this.updateChatChrome();
  }

  /**
   * Reflect the configured backend: enable/disable the Chat tab and show the
   * active model. Called on every mode switch and whenever settings change.
   */
  public updateChatChrome(): void {
    const canChat = this.chatAvailable();
    this.chatTab.toggleClass("is-disabled", !canChat);
    this.chatTab.toggleAttribute("disabled", !canChat);
    this.chatTab.setAttribute(
      "title",
      canChat ? "" : "Set a local (Ollama / LM Studio) or hosted model in settings to chat",
    );
    if (!canChat && this.mode === "chat") {
      this.setMode("search");
      return;
    }
    if (this.mode === "chat") {
      this.modelEl.show();
      this.modelEl.setText(`Answering with: ${this.index.generationSummary}`);
    } else {
      this.modelEl.hide();
    }
  }

  /** Show/hide the "still indexing" warning to match the current index state. */
  private refreshIndexingNotice(): void {
    const indexing = this.index.isIndexing;
    this.indexingEl.toggle(indexing);
    if (indexing) {
      this.indexingEl.setText(
        "⏳ Still indexing your vault — results may be incomplete until indexing finishes.",
      );
    }
  }

  /**
   * Called by the plugin when background indexing starts or finishes. Refresh the
   * warning, and if a search is on screen re-run it so results fill in as the
   * index completes — otherwise an empty mid-index search reads as "no matches".
   */
  public onIndexingStateChanged(): void {
    this.refreshIndexingNotice();
    if (this.mode === "search" && this.inputEl.value.trim().length > 0) {
      void this.runSearch();
    }
  }

  private renderSearchEmpty(): void {
    this.searchBodyEl.empty();
    this.searchBodyEl.createDiv({
      cls: "vaultsleuth-empty",
      text: "Type to search your vault semantically.",
    });
  }

  private renderChatEmpty(): void {
    this.chatBodyEl.empty();
    this.chatBodyEl.createDiv({
      cls: "vaultsleuth-empty",
      text: "Ask anything — answers are grounded in your notes and cite their sources.",
    });
  }

  private submit(): Promise<void> {
    return this.mode === "chat" ? this.sendChat() : this.runSearch();
  }

  // --- Search mode ---------------------------------------------------------

  private scheduleSearch(): void {
    if (this.debounceTimer !== null) {
      window.clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = window.setTimeout(() => void this.runSearch(), 200);
  }

  private async runSearch(): Promise<void> {
    this.refreshIndexingNotice();
    const query = this.inputEl.value.trim();
    if (query.length === 0) {
      this.renderSearchEmpty();
      this.statusEl.setText("");
      return;
    }
    const seq = ++this.searchSeq;
    this.statusEl.setText("Searching…");
    let results;
    try {
      results = await this.index.search(query, "hybrid", DEFAULT_TOP_K);
    } catch (error) {
      if (seq !== this.searchSeq) {
        return;
      }
      // Most likely the embedding model failed to load (e.g. offline before the
      // one-time download). Surface it instead of leaving "Searching…" forever.
      this.statusEl.setText("");
      this.searchBodyEl.empty();
      this.searchBodyEl.createDiv({
        cls: "vaultsleuth-empty",
        text: `Search failed: ${error instanceof Error ? error.message : String(error)}`,
      });
      return;
    }
    // A newer search started while this one was awaiting — discard stale results.
    if (seq !== this.searchSeq) {
      return;
    }
    this.statusEl.setText(`${results.length} result${results.length === 1 ? "" : "s"}`);
    this.searchBodyEl.empty();
    if (results.length === 0) {
      this.searchBodyEl.createDiv({
        cls: "vaultsleuth-empty",
        text: this.index.isIndexing
          ? "No matches yet — your vault is still indexing. Results will appear as it finishes."
          : "No matches found.",
      });
      return;
    }
    for (const result of results) {
      this.renderResult(result);
    }
  }

  private renderResult(result: SearchResult): void {
    const card = this.searchBodyEl.createDiv({ cls: "vaultsleuth-result" });
    const head = card.createDiv({ cls: "vaultsleuth-result-header" });
    head.createDiv({ cls: "vaultsleuth-result-title", text: result.chunk.noteTitle });
    head.createDiv({ cls: "vaultsleuth-result-score", text: result.score.toFixed(3) });
    if (result.chunk.heading.length > 0) {
      card.createDiv({ cls: "vaultsleuth-result-snippet", text: `# ${result.chunk.heading}` });
    }
    card.createDiv({ cls: "vaultsleuth-result-snippet", text: result.snippet });

    const actions = card.createDiv({ cls: "vaultsleuth-result-actions" });
    this.addResultAction(actions, "Open in split", () => this.openInSplit(result.chunk.notePath));
    this.addResultAction(actions, "Insert link", () => this.insertLink(result.chunk.notePath));
    this.addResultAction(actions, "Copy citation", () => this.copyCitation(result.chunk.notePath));

    card.addEventListener("click", (event) => {
      if (!(event.target as HTMLElement).hasClass("vaultsleuth-result-action")) {
        void this.openInSplit(result.chunk.notePath);
      }
    });
  }

  private addResultAction(parent: HTMLElement, label: string, handler: () => void): void {
    const el = parent.createSpan({ cls: "vaultsleuth-result-action", text: label });
    el.addEventListener("click", (event) => {
      event.stopPropagation();
      handler();
    });
  }

  // --- Chat mode -----------------------------------------------------------

  private chat(): ChatEngine {
    if (this.chatEngine === null) {
      this.chatEngine = this.index.createChatEngine();
    }
    return this.chatEngine;
  }

  private resetConversation(): void {
    this.chatEngine = this.index.createChatEngine();
    this.updateChatChrome();
    this.renderChatEmpty();
    this.inputEl.focus();
  }

  private async sendChat(): Promise<void> {
    const text = this.inputEl.value.trim();
    if (text.length === 0 || this.sending) {
      return;
    }
    this.sending = true;
    this.inputEl.value = "";
    this.refreshIndexingNotice();
    if (this.chatBodyEl.querySelector(".vaultsleuth-empty") !== null) {
      this.chatBodyEl.empty();
    }
    this.renderUserMessage(text);
    const pending = this.chatBodyEl.createDiv({ cls: "vaultsleuth-chat-message is-assistant" });
    pending.createDiv({ cls: "vaultsleuth-chat-bubble is-pending", text: "Thinking…" });
    this.scrollToBottom();

    try {
      const reply = await this.chat().ask(text);
      pending.remove();
      await this.renderAssistant(reply.message, reply.context);
    } catch (error) {
      // A thrown error is a backend/transport failure (server down, bad config) —
      // not a content refusal. Show it as a distinct error, not an answer bubble.
      pending.remove();
      const detail = error instanceof Error ? error.message : String(error);
      const el = this.chatBodyEl.createDiv({ cls: "vaultsleuth-chat-message is-assistant" });
      el.createDiv({
        cls: "vaultsleuth-chat-error",
        text: `Couldn't reach the model (${detail}). Check that your ${this.index.generationSummary} backend is running and configured in settings.`,
      });
    } finally {
      this.sending = false;
      this.scrollToBottom();
    }
  }

  private renderUserMessage(text: string): void {
    const el = this.chatBodyEl.createDiv({ cls: "vaultsleuth-chat-message is-user" });
    el.createDiv({ cls: "vaultsleuth-chat-bubble", text });
    this.addCopyAction(el, () => text);
  }

  /** Append a "Copy" control that writes the message text to the clipboard. */
  private addCopyAction(parent: HTMLElement, getText: () => string): void {
    const actions = parent.createDiv({ cls: "vaultsleuth-msg-actions" });
    const copy = actions.createSpan({ cls: "vaultsleuth-msg-action", text: "Copy" });
    copy.addEventListener("click", async () => {
      await navigator.clipboard.writeText(getText());
      new Notice("Copied");
    });
  }

  private async renderAssistant(message: ChatMessage, context: SearchResult[]): Promise<void> {
    const el = this.chatBodyEl.createDiv({ cls: "vaultsleuth-chat-message is-assistant" });
    const bubble = el.createDiv({ cls: "vaultsleuth-chat-bubble" });
    bubble.toggleClass("is-refusal", message.refused);
    await MarkdownRenderer.render(this.app, message.content, bubble, "", this);
    this.renderContext(el, context);
    if (!message.refused && message.grounded === false) {
      el.createDiv({
        cls: "vaultsleuth-chat-grounding",
        text: "⚠ Weak match in your notes — this answer may include the model's general knowledge.",
      });
    }
    this.addCopyAction(el, () => message.content);
  }

  /** Show the notes retrieved by semantic search and fed to the model as context. */
  private renderContext(parent: HTMLElement, context: SearchResult[]): void {
    const notes = [...new Set(context.map((result) => result.chunk.notePath))];
    if (notes.length === 0) {
      return;
    }
    const wrap = parent.createDiv({ cls: "vaultsleuth-context" });
    wrap.createSpan({ cls: "vaultsleuth-context-label", text: "Context used:" });
    for (const path of notes) {
      const chip = wrap.createSpan({ cls: "vaultsleuth-context-chip", text: noteBasename(path) });
      chip.addEventListener("click", () => void this.openInSplit(path));
    }
  }

  private scrollToBottom(): void {
    this.chatBodyEl.scrollTo({ top: this.chatBodyEl.scrollHeight });
  }

  // --- Shared note actions -------------------------------------------------

  private getFile(path: string): TFile | null {
    const file = this.app.vault.getAbstractFileByPath(path);
    return file instanceof TFile ? file : null;
  }

  private async openInSplit(path: string): Promise<void> {
    const file = this.getFile(path);
    if (file === null) {
      new Notice(`Note not found — ${path}`);
      return;
    }
    await this.app.workspace.getLeaf("split").openFile(file);
  }

  private insertLink(path: string): void {
    const markdownView = this.app.workspace.activeEditor;
    const link = `[[${noteBasename(path)}]]`;
    if (markdownView?.editor) {
      markdownView.editor.replaceSelection(link);
      new Notice("Link inserted");
    } else {
      new Notice("Open a note to insert a link");
    }
  }

  private async copyCitation(path: string): Promise<void> {
    await navigator.clipboard.writeText(`[[${noteBasename(path)}]]`);
    new Notice("Citation copied");
  }
}
