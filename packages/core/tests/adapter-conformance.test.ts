import { describe, expect, test } from "bun:test";
import { IntentAdapter } from "../src/adapters/IntentAdapter";
import { ProjectionAdapter } from "../src/adapters/ProjectionAdapter";
import { AlignmentAdapter } from "../src/adapters/AlignmentAdapter";
import { LoraIntentAdapter } from "../src/adapters/LoraIntentAdapter";
import type { WarpAdapter, AdapterState } from "../src/interfaces/WarpAdapter";

// Conformance test utility to verify the AdapterState interface contract
function testAdapterConformance(
  adapterName: string,
  adapter: WarpAdapter,
  testVector: number[],
  tuneMethodName: string = "tune",
  tuneArgs: any[] = [],
) {
  describe(`${adapterName} Conformance`, () => {
    test("exportState() returns a valid AdapterState object", () => {
      const state = adapter.exportState();

      // 1. Must be an object, not a string or array
      expect(typeof state).toBe("object");
      expect(state).not.toBeNull();
      expect(Array.isArray(state)).toBeFalse();

      // 2. Must be JSON serializable without losing type information
      const serialized = JSON.stringify(state);
      const deserialized = JSON.parse(serialized);
      expect(deserialized).toEqual(state);
    });

    test("importState() correctly restores the adapter", () => {
      const state = adapter.exportState();
      const AdapterClass = adapter.constructor as any;

      if (typeof AdapterClass.importState !== "function") {
        throw new Error(`${adapterName} missing static importState method`);
      }

      const restored = AdapterClass.importState(state) as WarpAdapter;

      // 3. The restored adapter must produce identical output to the original adapter
      const originalOutput = (adapter as any)[tuneMethodName](
        testVector,
        ...tuneArgs,
      );
      const restoredOutput = (restored as any)[tuneMethodName](
        testVector,
        ...tuneArgs,
      );

      expect(restoredOutput).toEqual(originalOutput);
    });
  });
}

describe("Adapter Conformance Tests (Core)", () => {
  const dim = 4;
  const testVector = [1.0, -0.5, 2.0, -1.5];

  const intentAdapter = new IntentAdapter(dim);
  intentAdapter.addIntent("default", {
    matrix: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    bias: [0, 0, 0, 0],
  });

  const projectionAdapter = new ProjectionAdapter(4, 2);
  projectionAdapter.addProjection("default", {
    matrix: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ],
    bias: [0, 0],
  });

  const alignmentAdapter = new AlignmentAdapter(4, 4);
  alignmentAdapter.addProjection("default", {
    matrix: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    bias: [0, 0, 0, 0],
  });

  const loraIntentAdapter = new LoraIntentAdapter(dim, 2);
  loraIntentAdapter.addIntent("default", {
    matrixA: [
      [1, 0],
      [0, 1],
      [0, 0],
      [0, 0],
    ],
    matrixB: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ],
    bias: [0, 0, 0, 0],
  });

  testAdapterConformance("IntentAdapter", intentAdapter, testVector, "tune", [
    "default",
  ]);
  testAdapterConformance(
    "ProjectionAdapter",
    projectionAdapter,
    testVector,
    "tune",
    ["default"],
  );
  testAdapterConformance(
    "AlignmentAdapter",
    alignmentAdapter,
    testVector,
    "tune",
    ["default"],
  );
  testAdapterConformance(
    "LoraIntentAdapter",
    loraIntentAdapter,
    testVector,
    "tune",
    ["default"],
  );
});
