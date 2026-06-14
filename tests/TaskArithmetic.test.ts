import { describe, expect, test } from "bun:test";
import { TaskArithmetic } from "../src/TaskArithmetic";
import { IntentWeights } from "../src/IntentAdapter";

describe("Task Arithmetic (Model Merging)", () => {
  const dim = 2;

  test("merge without baseIntent calculates W_new = I + scale * (W_task - I)", () => {
    const task1: IntentWeights = {
      matrix: [
        [2, 0],
        [0, 2],
      ],
      bias: [1, 1],
    };

    // scale = 0.5
    // base is Identity: W=[1,0; 0,1], b=[0,0]
    // dW = W_task - I = [1,0; 0,1]
    // db = [1,1]
    // W_new = I + 0.5 * dW = [1.5, 0; 0, 1.5]
    // b_new = 0 + 0.5 * db = [0.5, 0.5]

    const merged = TaskArithmetic.merge([{ weights: task1, scale: 0.5 }]);

    expect(merged.matrix[0]).toBeCloseTo(1.5);
    expect(merged.matrix[1]).toBeCloseTo(0);
    expect(merged.matrix[2]).toBeCloseTo(0);
    expect(merged.matrix[3]).toBeCloseTo(1.5);

    expect(merged.bias[0]).toBeCloseTo(0.5);
    expect(merged.bias[1]).toBeCloseTo(0.5);
  });

  test("merge with baseIntent combines multiple tasks correctly", () => {
    const base: IntentWeights = {
      matrix: [
        [1, 2],
        [3, 4],
      ],
      bias: [1, -1],
    };

    const task1: IntentWeights = {
      matrix: [
        [2, 2],
        [3, 5],
      ],
      bias: [2, 0], // db = [1, 1]
    };

    const task2: IntentWeights = {
      matrix: [
        [1, 0], // dW = [0, -2]
        [0, 4], // dW = [-3, 0]
      ],
      bias: [1, -3], // db = [0, -2]
    };

    // scale1 = 1.0, scale2 = 0.5
    // W_new = base + 1.0 * ([1,0; 0,1]) + 0.5 * ([0,-2; -3,0])
    //       = [1,2; 3,4] + [1,0; 0,1] + [0,-1; -1.5,0]
    //       = [2, 1; 1.5, 5]
    // b_new = base + 1.0 * [1,1] + 0.5 * [0,-2]
    //       = [1,-1] + [1,1] + [0,-1]
    //       = [2, -1]

    const merged = TaskArithmetic.merge(
      [
        { weights: task1, scale: 1.0 },
        { weights: task2, scale: 0.5 },
      ],
      base,
    );

    expect(merged.matrix[0]).toBeCloseTo(2);
    expect(merged.matrix[1]).toBeCloseTo(1);
    expect(merged.matrix[2]).toBeCloseTo(1.5);
    expect(merged.matrix[3]).toBeCloseTo(5);

    expect(merged.bias[0]).toBeCloseTo(2);
    expect(merged.bias[1]).toBeCloseTo(-1);
  });

  test("merge throws error if no tasks are provided", () => {
    expect(() => TaskArithmetic.merge([])).toThrow();
  });
});
