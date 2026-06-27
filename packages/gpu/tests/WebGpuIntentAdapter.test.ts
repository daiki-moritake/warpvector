import { describe, it, expect } from "bun:test";
import { WebGpuIntentAdapter } from "../src/WebGpuIntentAdapter";
import { InputVector } from "@warpvector/core";

describe("WebGpuIntentAdapter", () => {
  it("should initialize if WebGPU is supported, otherwise warn", async () => {
    const adapter = new WebGpuIntentAdapter(
      { default: { matrix: [[1, 0], [0, 1]], bias: [0, 0] } },
      2,
      2
    );
    await adapter.init();
    
    // In CI / Node / Bun environments, navigator.gpu might be undefined.
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      // It should be able to run
      const vectors: InputVector[] = [[1, 2], [3, 4]];
      const results = await adapter.tuneBatchAsync!(vectors, "default");
      expect(results.length).toBe(2);
    } else {
      // Should throw or fail gracefully when trying to run without WebGPU
      const vectors: InputVector[] = [[1, 2], [3, 4]];
      let error = null;
      try {
        await adapter.tuneBatchAsync!(vectors, "default");
      } catch (e: any) {
        error = e;
      }
      expect(error).not.toBeNull();
      expect(error.message).toContain("WebGPU is not initialized or not supported");
    }
  });

  it("exportState and importState should preserve intent data", () => {
    const intents = { test: { matrix: [[2, 2]], bias: [1] } };
    const adapter = new WebGpuIntentAdapter(intents, 2, 1);
    
    const state = adapter.exportState() as any;
    expect(state.inputDim).toBe(2);
    expect(state.outputDim).toBe(1);
    expect(state.intents).toEqual(intents);

    const imported = WebGpuIntentAdapter.importState(state);
    const importedState = imported.exportState() as any;
    expect(importedState).toEqual(state);
  });
});
