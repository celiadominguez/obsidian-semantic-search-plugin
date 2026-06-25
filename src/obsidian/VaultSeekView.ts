/**
 * The unified VaultSeek panel: a single sidebar view with a Search ⇄ Chat toggle
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
export const VAULTSEEK_VIEW_TYPE = "vaultseek-view";

/** Which mode the shared panel is showing. */
export type ViewMode = "search" | "chat";

export class VaultSeekView extends ItemView {
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
  private searchBodyEl!: HTMLElement;
  private chatBodyEl!: HTMLElement;

  private debounceTimer: number | null = null;
  private sending = false;

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
    return "brain-circuit";
  }

  public async onOpen(): Promise<void> {
    const container = this.contentEl;
    container.empty();
    container.addClass("vaultseek-view");

    const header = container.createDiv({ cls: "vaultseek-header" });
    const tabs = header.createDiv({ cls: "vaultseek-tabs" });
    this.searchTab = tabs.createEl("button", { text: "Search", cls: "vaultseek-tab" });
    this.chatTab = tabs.createEl("button", { text: "Chat", cls: "vaultseek-tab" });
    this.resetButton = header.createEl("button", { text: "New chat", cls: "vaultseek-reset" });

    this.searchTab.addEventListener("click", () => this.setMode("search"));
    this.chatTab.addEventListener("click", () => {
      if (this.chatAvailable()) {
        this.setMode("chat");
      }
    });
    this.resetButton.addEventListener("click", () => this.resetConversation());

    const inputRow = container.createDiv({ cls: "vaultseek-search-row" });
    this.inputEl = inputRow.createEl("textarea", {
      cls: "vaultseek-input",
      attr: { rows: "1", placeholder: "Search your vault by meaning…" },
    });
    this.primaryButton = inputRow.createEl("button", { text: "Search", cls: "mod-cta" });

    this.modelEl = container.createDiv({ cls: "vaultseek-chat-model" });
    this.statusEl = container.createDiv({ cls: "vaultseek-status" });
    // Separate, persistent containers so switching modes never loses either side.
    this.searchBodyEl = container.createDiv({ cls: "vaultseek-body" });
    this.chatBodyEl = container.createDiv({ cls: "vaultseek-body" });

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
   * Reflect the configured backend: enable/disable the Chat tab and show the
   * active model. Called by the plugin when settings change so the tab updates
   * live, and on every mode switch.
   */
  public updateChatChrome(): void {
    const canChat = this.chatAvailable();
    this.chatTab.toggleClass("is-disabled", !canChat);
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

  private renderSearchEmpty(): void {
    this.searchBodyEl.empty();
    this.searchBodyEl.createDiv({
      cls: "vaultseek-empty",
      text: "Type to search your vault semantically.",
    });
  }

  private renderChatEmpty(): void {
    this.chatBodyEl.empty();
    this.chatBodyEl.createDiv({
      cls: "vaultseek-empty",
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
    const query = this.inputEl.value.trim();
    if (query.length === 0) {
      this.renderSearchEmpty();
      this.statusEl.setText("");
      return;
    }
    this.statusEl.setText("Searching…");
    const results = await this.index.search(query, "hybrid", DEFAULT_TOP_K);
    this.statusEl.setText(`${results.length} result${results.length === 1 ? "" : "s"}`);
    this.searchBodyEl.empty();
    if (results.length === 0) {
      this.searchBodyEl.createDiv({ cls: "vaultseek-empty", text: "No matches found." });
      return;
    }
    for (const result of results) {
      this.renderResult(result);
    }
  }

  private renderResult(result: SearchResult): void {
    const card = this.searchBodyEl.createDiv({ cls: "vaultseek-result" });
    const head = card.createDiv({ cls: "vaultseek-result-header" });
    head.createDiv({ cls: "vaultseek-result-title", text: result.chunk.noteTitle });
    head.createDiv({ cls: "vaultseek-result-score", text: result.score.toFixed(3) });
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
    if (this.chatBodyEl.querySelector(".vaultseek-empty") !== null) {
      this.chatBodyEl.empty();
    }
    this.renderUserMessage(text);
    const pending = this.chatBodyEl.createDiv({ cls: "vaultseek-chat-message is-assistant" });
    pending.createDiv({ cls: "vaultseek-chat-bubble is-pending", text: "Thinking…" });
    this.scrollToBottom();

    try {
      const reply = await this.chat().ask(text);
      pending.remove();
      await this.renderAssistant(reply.message, reply.context);
    } catch (error) {
      pending.remove();
      await this.renderAssistant(
        {
          role: "assistant",
          content: error instanceof Error ? error.message : String(error),
          citations: [],
          refused: true,
          grounded: false,
        },
        [],
      );
    } finally {
      this.sending = false;
      this.scrollToBottom();
    }
  }

  private renderUserMessage(text: string): void {
    const el = this.chatBodyEl.createDiv({ cls: "vaultseek-chat-message is-user" });
    el.createDiv({ cls: "vaultseek-chat-bubble", text });
    this.addCopyAction(el, () => text);
  }

  /** Append a "Copy" control that writes the message text to the clipboard. */
  private addCopyAction(parent: HTMLElement, getText: () => string): void {
    const actions = parent.createDiv({ cls: "vaultseek-msg-actions" });
    const copy = actions.createSpan({ cls: "vaultseek-msg-action", text: "Copy" });
    copy.addEventListener("click", async () => {
      await navigator.clipboard.writeText(getText());
      new Notice("VaultSeek: copied");
    });
  }

  private async renderAssistant(message: ChatMessage, context: SearchResult[]): Promise<void> {
    const el = this.chatBodyEl.createDiv({ cls: "vaultseek-chat-message is-assistant" });
    const bubble = el.createDiv({ cls: "vaultseek-chat-bubble" });
    bubble.toggleClass("is-refusal", message.refused);
    await MarkdownRenderer.render(this.app, message.content, bubble, "", this);
    this.renderContext(el, context);
    if (!message.refused && message.grounded === false) {
      el.createDiv({
        cls: "vaultseek-chat-grounding",
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
    const wrap = parent.createDiv({ cls: "vaultseek-context" });
    wrap.createSpan({ cls: "vaultseek-context-label", text: "Context used:" });
    for (const path of notes) {
      const chip = wrap.createSpan({ cls: "vaultseek-context-chip", text: noteBasename(path) });
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
