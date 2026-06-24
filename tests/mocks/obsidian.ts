/**
 * Minimal `obsidian` module stub for tests. The unit and acceptance suites
 * exercise the pure `core` layer and never construct these, but the alias keeps
 * any incidental import resolvable without pulling in the real runtime.
 */

export class Plugin {
  app: any;
  manifest: any;
  constructor(app?: any, manifest?: any) {
    this.app = app;
    this.manifest = manifest;
  }
}

export class PluginSettingTab {
  app: any;
  containerEl: any;
  constructor(app?: any) {
    this.app = app;
  }
}

export class ItemView {
  leaf: any;
  contentEl: any;
  app: any;
  constructor(leaf?: any) {
    this.leaf = leaf;
  }
}

export class Setting {
  constructor(_containerEl?: any) {}
  setName(): this {
    return this;
  }
  setDesc(): this {
    return this;
  }
  setHeading(): this {
    return this;
  }
  addText(): this {
    return this;
  }
  addTextArea(): this {
    return this;
  }
  addToggle(): this {
    return this;
  }
  addDropdown(): this {
    return this;
  }
  addSlider(): this {
    return this;
  }
}

export class Notice {
  constructor(_message?: string) {}
}

export class TFile {
  path = "";
  basename = "";
  extension = "md";
}

export const MarkdownRenderer = {
  async render(): Promise<void> {},
};

export async function requestUrl(): Promise<{ status: number; json: unknown }> {
  return { status: 200, json: {} };
}

export type WorkspaceLeaf = unknown;
export type App = unknown;
