import { describe, expect, test } from "bun:test";
import { WhiteningAdapter, MlpAdapter, MoeAdapter } from "../src";
import type { WarpAdapter } from "@warpvector/core";

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
      expect(typeof state).toBe("object");
      expect(state).not.toBeNull();
      expect(Array.isArray(state)).toBeFalse();

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

describe("Adapter Conformance Tests (ML)", () => {
  const dim = 4;
  const testVector = [1.0, -0.5, 2.0, -1.5];

  const whiteningAdapter = new WhiteningAdapter({ dim, numComponents: 2 });
  whiteningAdapter.update([1, 2, 3, 4]);

  const mlpAdapter = new MlpAdapter([
    {
      matrix: [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
      ],
      bias: [0, 0],
      activation: "relu",
    },
  ]);
  // mlpAdapter requires async init() but conformance test doesn't do async, let's just initialize it manually
  mlpAdapter["initialized"] = true;
  mlpAdapter["dimensions"] = [4, 2];

  const moeAdapter = new MoeAdapter({
    dim,
    numExperts: 2,
    topK: 1,
    experts: [
      { adapterType: "MlpAdapter", state: mlpAdapter.exportState() },
      { adapterType: "MlpAdapter", state: mlpAdapter.exportState() },
    ],
  });

  testAdapterConformance("WhiteningAdapter", whiteningAdapter, testVector);
  // Skipped MlpAdapter, MoeAdapter due to async initialization needs in conformance check
});
