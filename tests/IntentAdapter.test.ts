import { describe, expect, test } from "bun:test";
import { IntentAdapter, IntentWeights } from "../src/IntentAdapter";

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
});
