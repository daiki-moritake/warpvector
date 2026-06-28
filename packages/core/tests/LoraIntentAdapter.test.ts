import { describe, expect, test } from "bun:test";
import {
  LoraIntentAdapter,
  LoraIntentWeights,
} from "../src/adapters/LoraIntentAdapter";

describe("LoraIntentAdapter Core Logic", () => {
  const dummyIntents: Record<string, LoraIntentWeights> = {
    identity: {
      matrixA: [[0], [0], [0]],
      matrixB: [[0, 0, 0]],
      bias: [0, 0, 0],
    },
    // W = A*B = [[1],[1],[1]] * [[1,1,1]] = [[1,1,1],[1,1,1],[1,1,1]]
    scaleAndShift: {
      matrixA: [[1], [1], [1]],
      matrixB: [[1, 1, 1]],
      bias: [1, 2, 3],
    },
  };

  test("should correctly apply identity transformation (residual connection)", () => {
    // x' = x + A*B*x + b
    // A = 0, B = 0, b = 0 => x' = x
    const adapter = new LoraIntentAdapter(3, 1, dummyIntents);
    const base = [1, 2, 3];
    const result = adapter.tune(base, "identity");

    expect(result[0]).toBeCloseTo(1, 5);
    expect(result[1]).toBeCloseTo(2, 5);
    expect(result[2]).toBeCloseTo(3, 5);
  });

  test("should correctly apply loRA transformation", () => {
    // x' = x + A*B*x + b
    // B*x = [1,1,1]*[1,2,3]^T = 6
    // A*(6) = [6,6,6]^T
    // x' = [1,2,3] + [6,6,6] + [1,2,3] = [8, 10, 12]
    const adapter = new LoraIntentAdapter(3, 1, dummyIntents);
    const base = [1, 2, 3];
    const result = adapter.tune(base, "scaleAndShift");

    expect(result[0]).toBeCloseTo(8, 5);
    expect(result[1]).toBeCloseTo(10, 5);
    expect(result[2]).toBeCloseTo(12, 5);
  });

  test("should dynamically add and remove LoRA intents", () => {
    const adapter = new LoraIntentAdapter(3, 1, dummyIntents);
    adapter.addIntent("dynamic", {
      matrixA: [[2], [0], [0]],
      matrixB: [[1, 0, 0]],
      bias: [0, 0, 0],
    });

    // B*x = [1,0,0]*[5,5,5]^T = 5
    // A*5 = [10, 0, 0]^T
    // x' = [5,5,5] + [10,0,0] + [0,0,0] = [15, 5, 5]
    const result = adapter.tune([5, 5, 5], "dynamic");
    expect(result[0]).toBeCloseTo(15, 5);
    expect(result[1]).toBeCloseTo(5, 5);
    expect(result[2]).toBeCloseTo(5, 5);

    adapter.removeIntent("dynamic");
    expect(() => {
      adapter.tune([5, 5, 5], "dynamic");
    }).toThrow("Intent 'dynamic' not found.");
  });

  test("exportState and importState work correctly and states can be fed back into constructor", () => {
    const adapter = new LoraIntentAdapter(3, 1, dummyIntents);
    const state = adapter.exportState();
    expect(typeof state).toBe("string");

    // 1. importState で復元
    const restored = LoraIntentAdapter.importState(state);
    const result1 = restored.tune([1, 2, 3], "scaleAndShift");
    expect(result1[0]).toBeCloseTo(8, 5);

    // 2. 復元した adapter.exportState() で得た intents から新規に adapter を構築できるか検証
    const parsed = JSON.parse(state);
    expect(() => {
      new LoraIntentAdapter(parsed.dimension, parsed.rank, parsed.intents);
    }).not.toThrow();

    const reconstructed = new LoraIntentAdapter(parsed.dimension, parsed.rank, parsed.intents);
    const result2 = reconstructed.tune([1, 2, 3], "scaleAndShift");
    expect(result2[0]).toBeCloseTo(8, 5);
  });
});
