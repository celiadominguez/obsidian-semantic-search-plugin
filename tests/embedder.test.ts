import { describe, expect, it } from "vitest";
import { modelFileRelativePath } from "../src/core/embedder";

const MODEL = "Xenova/bge-small-en-v1.5";

describe("modelFileRelativePath", () => {
  it("maps a Hugging Face resolve URL to an on-disk relative path", () => {
    const url = `https://huggingface.co/${MODEL}/resolve/main/onnx/model_quantized.onnx`;
    expect(modelFileRelativePath(url, MODEL)).toBe(`${MODEL}/onnx/model_quantized.onnx`);
  });

  it("maps a config/tokenizer file too, stripping the resolve/<revision> infix", () => {
    expect(
      modelFileRelativePath(`https://huggingface.co/${MODEL}/resolve/abc123/config.json`, MODEL),
    ).toBe(`${MODEL}/config.json`);
  });

  it("handles a local-path style URL with no resolve/ infix", () => {
    expect(modelFileRelativePath(`models/${MODEL}/tokenizer.json`, MODEL)).toBe(
      `${MODEL}/tokenizer.json`,
    );
  });

  it("returns undefined for a URL that is not for this model", () => {
    expect(
      modelFileRelativePath("https://huggingface.co/some/other-model/config.json", MODEL),
    ).toBeUndefined();
  });
});
