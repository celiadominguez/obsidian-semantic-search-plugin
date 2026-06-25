/**
 * Ambient type for `.wasm` imports. The esbuild build configures a `binary`
 * loader for `.wasm`, so importing one yields its bytes as a `Uint8Array` rather
 * than a URL. Used to inline the ONNX Runtime WASM engine into `main.js`.
 */
declare module "*.wasm" {
  const bytes: Uint8Array;
  export default bytes;
}
