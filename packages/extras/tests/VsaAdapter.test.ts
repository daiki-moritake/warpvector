import { describe, expect, test } from "bun:test";
import { VsaAdapter } from "../src/adapters/VsaAdapter";

describe("Vector Symbolic Architecture (VSA) Adapter", () => {
  test("bundle correctly adds and normalizes vectors", () => {
    const v1 = [1, 0, 0];
    const v2 = [0, 1, 0];

    // bundle without normalize
    const resUnnorm = VsaAdapter.bundle([v1, v2], { shouldNormalize: false });
    expect(resUnnorm[0]).toBeCloseTo(1);
    expect(resUnnorm[1]).toBeCloseTo(1);
    expect(resUnnorm[2]).toBeCloseTo(0);

    // bundle with normalize
    const resNorm = VsaAdapter.bundle([v1, v2], { shouldNormalize: true });
    // norm of [1,1,0] is sqrt(2) approx 1.414
    expect(resNorm[0]).toBeCloseTo(1 / Math.sqrt(2));
    expect(resNorm[1]).toBeCloseTo(1 / Math.sqrt(2));
    expect(resNorm[2]).toBeCloseTo(0);
  });

  test("bind correctly multiplies vectors element-wise", () => {
    const v1 = [2, 3, 4];
    const v2 = [0.5, 2, 0];

    const bound = VsaAdapter.bind(v1, v2, { shouldNormalize: false });
    expect(bound[0]).toBeCloseTo(1);
    expect(bound[1]).toBeCloseTo(6);
    expect(bound[2]).toBeCloseTo(0);
  });

  test("unbind correctly divides vectors element-wise to retrieve original", () => {
    const v1 = [2, 3, 4];
    const key = [0.5, 2, 0.5]; // avoid zero for precise unbinding test

    const bound = VsaAdapter.bind(v1, key, { shouldNormalize: false });
    const unbound = VsaAdapter.unbind(bound, key, { shouldNormalize: false });

    expect(unbound[0]).toBeCloseTo(2);
    expect(unbound[1]).toBeCloseTo(3);
    expect(unbound[2]).toBeCloseTo(4);
  });

  test("unbind handles zero division safely", () => {
    const bound = [1, 1, 1];
    const key = [0, 1, 0];

    const unbound = VsaAdapter.unbind(bound, key, { shouldNormalize: false });
    expect(unbound[0]).toBeCloseTo(100000); // 1 / 1e-5
    expect(unbound[1]).toBeCloseTo(1);
    expect(unbound[2]).toBeCloseTo(100000);
  });

  test("bundle throws error if array is empty", () => {
    expect(() => VsaAdapter.bundle([])).toThrow();
  });

  describe("Binary VSA (1-bit Vector Symbolic Architecture)", () => {
    test("bindBinary and unbindBinary using XOR", () => {
      // 0b10101010 = 170
      // 0b11001100 = 204
      const v1 = new Uint8Array([170, 255, 0]);
      const v2 = new Uint8Array([204, 15, 240]);

      const bound = VsaAdapter.bindBinary(v1, v2);
      // 170 ^ 204 = 102 (0b01100110)
      // 255 ^ 15 = 240
      // 0 ^ 240 = 240
      expect(bound[0]).toBe(102);
      expect(bound[1]).toBe(240);
      expect(bound[2]).toBe(240);

      const unbound = VsaAdapter.unbindBinary(bound, v2);
      expect(unbound[0]).toBe(170);
      expect(unbound[1]).toBe(255);
      expect(unbound[2]).toBe(0);
    });

    test("bundleBinary correctly computes majority vote", () => {
      // We will test 3 vectors
      // Bit 0 (LSB): v1=1, v2=1, v3=0 => majority 1
      // Bit 1: v1=0, v2=0, v3=1 => majority 0
      // Bit 2: v1=1, v2=1, v3=1 => majority 1
      // Bit 3: v1=0, v2=0, v3=0 => majority 0
      // Let's construct bytes:
      // v1: 0b00000101 = 5
      // v2: 0b00000101 = 5
      // v3: 0b00000010 = 2
      // Majority should be: 0b00000101 = 5

      const v1 = new Uint8Array([5, 255]);
      const v2 = new Uint8Array([5, 0]);
      const v3 = new Uint8Array([2, 0]);

      const bundled = VsaAdapter.bundleBinary([v1, v2, v3]);
      expect(bundled[0]).toBe(5);
      // For the second byte: 255, 0, 0 => majority 0s
      expect(bundled[1]).toBe(0);
    });

    test("bundleBinary resolves ties to 1", () => {
      // Tie breaking: 2 vectors
      // v1: 0b00000001 (1)
      // v2: 0b00000000 (0)
      // Tie at LSB, should resolve to 1
      const v1 = new Uint8Array([1]);
      const v2 = new Uint8Array([0]);

      const bundled = VsaAdapter.bundleBinary([v1, v2]);
      expect(bundled[0]).toBe(1);
    });
  });
});
