import { describe, it, expect } from "bun:test";
import { WebGpuIntentAdapter } from "../src/WebGpuIntentAdapter";
import { InputVector } from "@warpvector/core";

describe("WebGpuIntentAdapter", () => {
  it("should initialize if WebGPU is supported, otherwise warn", async () => {
    const adapter = new WebGpuIntentAdapter(
      {
        default: {
          matrix: [
            [1, 0],
            [0, 1],
          ],
          bias: [0, 0],
        },
      },
      2,
      2,
    );
    await adapter.init();

    // In CI / Node / Bun environments, navigator.gpu might be undefined.
    if (typeof navigator !== "undefined" && navigator.gpu) {
      // It should be able to run
      const vectors: InputVector[] = [
        [1, 2],
        [3, 4],
      ];
      const results = await adapter.tuneBatchAsync!(vectors, "default");
      expect(results.length).toBe(2);
    } else {
      // Should fallback gracefully to WASM/CPU without WebGPU
      const vectors: InputVector[] = [
        [1, 2],
        [3, 4],
      ];
      const results = await adapter.tuneBatchAsync!(vectors, "default");
      expect(results.length).toBe(2);
      expect(results[0][0]).toBeCloseTo(1, 5);
      expect(results[0][1]).toBeCloseTo(2, 5);
    }
  });

  it("should always use fallback for synchronous tune()", () => {
    const adapter = new WebGpuIntentAdapter(
      {
        default: {
          matrix: [
            [1, 0],
            [0, 1],
          ],
          bias: [0, 0],
        },
      },
      2,
      2,
    );
    const vector: InputVector = [1, 2];
    const result = adapter.tune(vector, "default");
    expect(result[0]).toBeCloseTo(1, 5);
    expect(result[1]).toBeCloseTo(2, 5);
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
