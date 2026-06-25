/**
 * The ONNX Runtime WASM-CPU engine, inlined into `main.js` at build time via
 * esbuild's `binary` loader.
 *
 * Why inline rather than ship a separate file: Obsidian's community-plugin
 * installer only delivers `main.js`, `manifest.json`, and `styles.css` to a
 * user's vault — it does not download arbitrary release assets. A standalone
 * `.wasm` would therefore never reach users, and onnxruntime-web would fall back
 * to fetching the runtime from a CDN at load time, which is exactly the "download
 * remote code" pattern Obsidian prohibits. Inlining bakes the engine into the one
 * file that is always delivered, so nothing is fetched at runtime.
 *
 * This is the non-JSEP build (CPU only); the esbuild config aliases transformers'
 * onnxruntime-web import to the WASM-only ORT bundle, whose JS glue matches these
 * bytes.
 */
import wasmBytes from "onnxruntime-web/ort-wasm-simd-threaded.wasm";

/** The inlined ORT WASM bytes, for onnxruntime-web's `env.wasm.wasmBinary`. */
export function ortWasmBinary(): ArrayBuffer {
  return wasmBytes.buffer as ArrayBuffer;
}
