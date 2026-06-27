// @warpvector/core - Core vector space transformation engine
export * from "./interfaces/WarpAdapter";
export * from "./adapters/IntentAdapter";
export * from "./adapters/LoraIntentAdapter";
export * from "./adapters/ProjectionAdapter";
export * from "./pipeline/WarpPipeline";
export * from "./pipeline/AdapterRegistry";
export * from "./pipeline/FormatRegistry";
export * from "./adapters/VectorDBAdapter";
export * from "./errors";
export * from "./metrics";
export * from "./utils";
export * from "./validation";
export * from "./telemetry/WarpTracer";
export * from "./math/vector";

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
  getWasmMemoryStats,
  WasmMutex,
  wasmMutex,
  withWasmMemoryStack,
} from "./wasm/wasm-loader";
export type { WasmMemoryStats } from "./wasm/wasm-loader";
