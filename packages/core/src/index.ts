// @warpvector/core - Core vector space transformation engine
export * from "./interfaces/WarpAdapter";
export * from "./adapters/AbstractWarpAdapter";
export * from "./interfaces/training";
export * from "./adapters/IntentAdapter";
export * from "./adapters/LoraIntentAdapter";
export * from "./adapters/ProjectionAdapter";
export * from "./adapters/AlignmentAdapter";
export * from "./pipeline/WarpPipeline";
export * from "./pipeline/AdapterRegistry";
export * from "./pipeline/FormatRegistry";
export * from "./formatters/VectorDBFormatter";
export * from "./errors";
export * from "./metrics";
export * from "./utils";
export * from "./validation";
export * from "./telemetry/WarpTracer";
export * from "./math/vector";
export * from "./feedback/FeedbackCollector";
export * from "./feedback/AdaptiveScheduler";
export * from "./feedback/FederatedAggregator";

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
export { globalWasmPool } from "./wasm/WasmPool";
