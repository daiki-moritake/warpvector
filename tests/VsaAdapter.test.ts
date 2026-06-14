import { describe, expect, test } from "bun:test";
import { VsaAdapter } from "../src/VsaAdapter";

describe("Vector Symbolic Architecture (VSA) Adapter", () => {
  test("bundle correctly adds and normalizes vectors", () => {
    const v1 = [1, 0, 0];
    const v2 = [0, 1, 0];

    // bundle without normalize
    const resUnnorm = VsaAdapter.bundle([v1, v2], false);
    expect(resUnnorm[0]).toBeCloseTo(1);
    expect(resUnnorm[1]).toBeCloseTo(1);
    expect(resUnnorm[2]).toBeCloseTo(0);

    // bundle with normalize
    const resNorm = VsaAdapter.bundle([v1, v2], true);
    // norm of [1,1,0] is sqrt(2) approx 1.414
    expect(resNorm[0]).toBeCloseTo(1 / Math.sqrt(2));
    expect(resNorm[1]).toBeCloseTo(1 / Math.sqrt(2));
    expect(resNorm[2]).toBeCloseTo(0);
  });

  test("bind correctly multiplies vectors element-wise", () => {
    const v1 = [2, 3, 4];
    const v2 = [0.5, 2, 0];

    const bound = VsaAdapter.bind(v1, v2, false);
    expect(bound[0]).toBeCloseTo(1);
    expect(bound[1]).toBeCloseTo(6);
    expect(bound[2]).toBeCloseTo(0);
  });

  test("unbind correctly divides vectors element-wise to retrieve original", () => {
    const v1 = [2, 3, 4];
    const key = [0.5, 2, 0.5]; // avoid zero for precise unbinding test

    const bound = VsaAdapter.bind(v1, key, false);
    const unbound = VsaAdapter.unbind(bound, key, false);

    expect(unbound[0]).toBeCloseTo(2);
    expect(unbound[1]).toBeCloseTo(3);
    expect(unbound[2]).toBeCloseTo(4);
  });

  test("unbind handles zero division safely", () => {
    const bound = [1, 1, 1];
    const key = [0, 1, 0];

    const unbound = VsaAdapter.unbind(bound, key, false);
    expect(unbound[0]).toBeGreaterThan(1000000); // 1 / 1e-8
    expect(unbound[1]).toBeCloseTo(1);
    expect(unbound[2]).toBeGreaterThan(1000000);
  });

  test("bundle throws error if array is empty", () => {
    expect(() => VsaAdapter.bundle([])).toThrow();
  });
});
