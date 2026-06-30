import { describe, expect, test } from "bun:test";
import { SoftWhiteningAdapter } from "../src/adapters/SoftWhiteningAdapter";
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

describe("Adapter Conformance Tests (Train)", () => {
  const dim = 4;
  const testVector = [1.0, -0.5, 2.0, -1.5];

  const softWhiteningAdapter = new SoftWhiteningAdapter({
    dim,
    numComponents: 2,
  });
  softWhiteningAdapter.update([1, 2, 3, 4]);

  testAdapterConformance(
    "SoftWhiteningAdapter",
    softWhiteningAdapter,
    testVector,
  );
});
