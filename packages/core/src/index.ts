// @warpvector/core - Core vector space transformation engine
export * from "./interfaces/WarpAdapter";
export * from "./adapters/IntentAdapter";
export * from "./adapters/LoraIntentAdapter";
export * from "./adapters/ProjectionAdapter";
export * from "./pipeline/WarpPipeline";
export * from "./adapters/VectorDBAdapter";
export * from "./utils";
export * from "./validation";

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
