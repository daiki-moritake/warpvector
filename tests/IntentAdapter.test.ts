import { describe, expect, test } from "bun:test";
import { IntentAdapter, IntentWeights } from "../src/IntentAdapter";
import { normalize } from "../src/utils";

describe("IntentAdapter Core Logic", () => {
  const dummyIntents: Record<string, IntentWeights> = {
    identity: {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ],
      bias: [0, 0, 0]
    },
    scaleAndShift: {
      matrix: [
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 0.5]
      ],
      bias: [1, -1, 2]
    },
    complexTransform: {
      matrix: [
        [0.5, -0.2, 0.1],
        [0.1,  0.8, -0.3],
        [-0.4, 0.5,  0.9]
      ],
      bias: [0.1, 0.2, -0.1]
    }
  };

  test("should correctly apply identity transformation", () => {
    const adapter = new IntentAdapter(dummyIntents);
    const base = [1.5, -2.0, 3.14];
    const result = adapter.tune(base, "identity");

    expect(result.length).toBe(3);
    expect(result[0]).toBeCloseTo(1.5, 5);
    expect(result[1]).toBeCloseTo(-2.0, 5);
    expect(result[2]).toBeCloseTo(3.14, 5);
  });

  test("should correctly apply scale and shift transformation", () => {
    const adapter = new IntentAdapter(dummyIntents);
    const base = [2, 4, 8];
    const result = adapter.tune(base, "scaleAndShift");

    // Expected:
    // x: 2 * 2 + 1 = 5
    // y: 3 * 4 - 1 = 11
    // z: 0.5 * 8 + 2 = 6
    expect(result[0]).toBeCloseTo(5, 5);
    expect(result[1]).toBeCloseTo(11, 5);
    expect(result[2]).toBeCloseTo(6, 5);
  });

  test("should correctly apply complex affine transformation", () => {
    const adapter = new IntentAdapter(dummyIntents);
    const base = new Float32Array([1, 2, 3]);
    const result = adapter.tune(base, "complexTransform");

    // Expected:
    // x: (0.5 * 1) + (-0.2 * 2) + (0.1 * 3) + 0.1 = 0.5 - 0.4 + 0.3 + 0.1 = 0.5
    // y: (0.1 * 1) + (0.8 * 2) + (-0.3 * 3) + 0.2 = 0.1 + 1.6 - 0.9 + 0.2 = 1.0
    // z: (-0.4 * 1) + (0.5 * 2) + (0.9 * 3) - 0.1 = -0.4 + 1.0 + 2.7 - 0.1 = 3.2
    expect(result[0]).toBeCloseTo(0.5, 5);
    expect(result[1]).toBeCloseTo(1.0, 5);
    expect(result[2]).toBeCloseTo(3.2, 5);
  });

  test("should throw error if an unknown intent is requested", () => {
    const adapter = new IntentAdapter(dummyIntents);
    expect(() => {
      adapter.tune([1, 2, 3], "unknownIntent");
    }).toThrow("Intent 'unknownIntent' not found.");
  });

  test("should throw error if base vector dimension is mismatched", () => {
    const adapter = new IntentAdapter(dummyIntents);
    expect(() => {
      adapter.tune([1, 2], "identity"); // Missing 1 dimension
    }).toThrow("Vector dimension mismatch. Expected 3, got 2.");
  });

  test("should throw error on initialization if dimensions are inconsistent", () => {
    const invalidIntents: Record<string, IntentWeights> = {
      badIntent: {
        matrix: [
          [1, 0],
          [0, 1]
        ],
        bias: [0, 0, 0] // 3D bias, 2D matrix
      }
    };

    expect(() => {
      new IntentAdapter(invalidIntents);
    }).toThrow("Intent 'badIntent': Matrix row dimension mismatch. Expected 3, got 2.");
  });

  test("バッチ処理(tuneBatch)が正しく適用されること", () => {
    const adapter = new IntentAdapter(dummyIntents);
    const base1 = [2, 4, 8];
    const base2 = new Float32Array([1, 1, 1]);
    const results = adapter.tuneBatch([base1, base2], "scaleAndShift");

    expect(results.length).toBe(2);
    expect(results[0][0]).toBeCloseTo(5, 5);
    expect(results[0][1]).toBeCloseTo(11, 5);
    expect(results[0][2]).toBeCloseTo(6, 5);
    expect(results[1][0]).toBeCloseTo(3, 5);
    expect(results[1][1]).toBeCloseTo(2, 5);
    expect(results[1][2]).toBeCloseTo(2.5, 5);
  });

  test("should throw error on initialization if matrix is not square", () => {
    const invalidIntents: Record<string, IntentWeights> = {
      badIntent: {
        matrix: [
          [1, 0, 0],
          [0, 1], // Invalid column length
          [0, 0, 1]
        ],
        bias: [0, 0, 0]
      }
    };

    expect(() => {
      new IntentAdapter(invalidIntents);
    }).toThrow("Intent 'badIntent': Matrix column dimension mismatch at row 1. Expected 3, got 2.");
  });

  test("動的にインテントの追加と削除ができること", () => {
    const adapter = new IntentAdapter(dummyIntents);
    adapter.addIntent("dynamicIntent", {
      matrix: [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
      ],
      bias: [1, 2, 3]
    });

    const result1 = adapter.tune([1, 0, 0], "dynamicIntent");
    expect(result1[0]).toBeCloseTo(2, 5); // 1*1 + 1 = 2
    expect(result1[1]).toBeCloseTo(3, 5); // 1*1 + 2 = 3
    expect(result1[2]).toBeCloseTo(4, 5); // 1*1 + 3 = 4

    adapter.removeIntent("dynamicIntent");
    expect(() => {
      adapter.tune([1, 0, 0], "dynamicIntent");
    }).toThrow("Intent 'dynamicIntent' not found.");
  });

  test("tuneBlendedで複数のインテントをブレンドできること", () => {
    const adapter = new IntentAdapter(dummyIntents);
    const base = [2, 4, 8];
    // Blend: 0.5 * identity + 0.5 * scaleAndShift
    // identity: W*x+b = [2, 4, 8]
    // scaleAndShift: W*x+b = [5, 11, 6]
    // 0.5*[2, 4, 8] + 0.5*[5, 11, 6] = [3.5, 7.5, 7]
    const result = adapter.tuneBlended(base, {
      identity: 0.5,
      scaleAndShift: 0.5
    });

    expect(result[0]).toBeCloseTo(3.5, 5);
    expect(result[1]).toBeCloseTo(7.5, 5);
    expect(result[2]).toBeCloseTo(7.0, 5);
  });
});

describe("Utils", () => {
  test("normalizeが正しくベクトルを正規化できること", () => {
    const base = [3, 4];
    const result = normalize(base);
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
    
    const norm = Math.sqrt(result[0] * result[0] + result[1] * result[1]);
    expect(norm).toBeCloseTo(1.0, 5);
  });

  test("ゼロベクトルのnormalizeはゼロベクトルを返すこと", () => {
    const base = [0, 0, 0];
    const result = normalize(base);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(0);
    expect(result[2]).toBe(0);
  });
});
