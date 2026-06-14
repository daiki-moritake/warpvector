// @warpvector/core - Core vector space transformation engine
export * from "./WarpAdapter";
export * from "./IntentAdapter";
export * from "./LoraIntentAdapter";
export * from "./ProjectionAdapter";
export * from "./WarpPipeline";
export * from "./db";
export * from "./utils";

// WASM runtime utilities (used by sub-packages @warpvector/ml, @warpvector/extras)
export {
  initWasm,
  getWasmInstance,
  getWasmMemory,
  ensureWasmMemory,
  allocateWasmMemory,
  resetWasmAllocator,
  getWasmAllocatorOffset,
  setWasmAllocatorOffset,
  writeFloat32ArrayToWasm,
  readFloat32ArrayFromWasm,
  WasmMutex,
  wasmMutex,
  withWasmMemoryStack,
} from "./wasm/wasm-loader";
