import { describe, expect, test } from "bun:test";
import { ProjectionAdapter, ProjectionWeights } from "../src/ProjectionAdapter";

describe("ProjectionAdapter Core Logic", () => {
  const dummyProjections: Record<string, ProjectionWeights> = {
    // 3次元から2次元への射影
    reduceTo2D: {
      matrix: [
        [1, 0, 0], // x軸のみ
        [0, 1, 0], // y軸のみ
      ],
    },
  };

  test("should correctly apply dimensionality reduction", () => {
    // 3次元 -> 2次元
    const adapter = new ProjectionAdapter(3, 2, dummyProjections);
    const base = [5, 10, 15];

    // W * x = [[1,0,0], [0,1,0]] * [5,10,15]^T = [5, 10]^T
    const result = adapter.project(base, "reduceTo2D");

    expect(result.length).toBe(2);
    expect(result[0]).toBeCloseTo(5, 5);
    expect(result[1]).toBeCloseTo(10, 5);
  });

  test("should throw error if matrix size is invalid", () => {
    expect(() => {
      new ProjectionAdapter(3, 2, {
        invalid: {
          matrix: [
            [1, 0],
            [0, 1],
          ], // 2x2 instead of 2x3
        },
      });
    }).toThrow();
  });

  test("should dynamically add and remove projections", () => {
    const adapter = new ProjectionAdapter(3, 1);
    adapter.addProjection("reduceTo1D", {
      matrix: [[1, 1, 1]], // 合計値への射影
    });

    const result = adapter.project([2, 3, 4], "reduceTo1D");
    expect(result.length).toBe(1);
    expect(result[0]).toBeCloseTo(9, 5);

    adapter.removeProjection("reduceTo1D");
    expect(() => {
      adapter.project([2, 3, 4], "reduceTo1D");
    }).toThrow("Projection 'reduceTo1D' not found.");
  });

  test("should throw error if input dimension is mismatched", () => {
    const adapter = new ProjectionAdapter(3, 2, dummyProjections);
    expect(() => {
      adapter.project([1, 2], "reduceTo2D"); // 2次元を与えてしまう
    }).toThrow();
  });
});
