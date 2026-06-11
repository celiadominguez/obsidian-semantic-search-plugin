/**
 * The build (see `esbuild.config.mjs`) bundles the embedding Web Worker into a
 * string and exposes it through this virtual module, so the main thread can
 * instantiate the worker from a Blob URL without shipping a sidecar file.
 */
declare module "inline:embed-worker" {
  const workerCode: string;
  export default workerCode;
}
