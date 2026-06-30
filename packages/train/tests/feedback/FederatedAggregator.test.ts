import { describe, it, expect } from "bun:test";
import { FederatedAggregator } from "@warpvector/core";
import type { IntentWeights } from "@warpvector/core";

const DIM = 2;

function makeIdentityWeights(): IntentWeights {
  // 2x2 identity matrix
  return {
    matrix: new Float32Array([1, 0, 0, 1]),
    bias: new Float32Array([0, 0]),
  };
}

describe("FederatedAggregator", () => {
  it("single client returns client weights", () => {
    const base = makeIdentityWeights();
    const agg = new FederatedAggregator(base, DIM);

    const clientWeights: IntentWeights = {
      matrix: new Float32Array([1.1, 0.2, 0.3, 1.4]),
      bias: new Float32Array([0.5, 0.6]),
    };

    agg.submitUpdate({ weights: clientWeights, interactionCount: 10 });

    const result = agg.aggregate();

    // 1クライアントの場合: W_new = W_base + 1.0 * (W_client - W_base) = W_client
    expect(result.matrix[0]).toBeCloseTo(1.1, 5);
    expect(result.matrix[1]).toBeCloseTo(0.2, 5);
    expect(result.matrix[2]).toBeCloseTo(0.3, 5);
    expect(result.matrix[3]).toBeCloseTo(1.4, 5);
    expect(result.bias[0]).toBeCloseTo(0.5, 5);
    expect(result.bias[1]).toBeCloseTo(0.6, 5);
  });

  it("two clients with equal counts averages deltas", () => {
    const base = makeIdentityWeights();
    const agg = new FederatedAggregator(base, DIM);

    // Client A: matrix[0] を +0.2 変更
    const clientA: IntentWeights = {
      matrix: new Float32Array([1.2, 0, 0, 1]),
      bias: new Float32Array([0, 0]),
    };

    // Client B: matrix[0] を +0.4 変更
    const clientB: IntentWeights = {
      matrix: new Float32Array([1.4, 0, 0, 1]),
      bias: new Float32Array([0, 0]),
    };

    agg.submitUpdate({ weights: clientA, interactionCount: 10 });
    agg.submitUpdate({ weights: clientB, interactionCount: 10 });

    const result = agg.aggregate();

    // W_new = I + 0.5*(A-I) + 0.5*(B-I) = I + 0.5*0.2 + 0.5*0.4 = 1.0 + 0.3 = 1.3
    expect(result.matrix[0]).toBeCloseTo(1.3, 5);
    expect(result.matrix[3]).toBeCloseTo(1.0, 5); // 未変更部分
  });

  it("weights by interactionCount", () => {
    const base = makeIdentityWeights();
    const agg = new FederatedAggregator(base, DIM);

    // Client A: delta = +0.1, count = 100 (多い)
    const clientA: IntentWeights = {
      matrix: new Float32Array([1.1, 0, 0, 1]),
      bias: new Float32Array([0, 0]),
    };

    // Client B: delta = +0.5, count = 100 (多い)
    const clientB: IntentWeights = {
      matrix: new Float32Array([1.5, 0, 0, 1]),
      bias: new Float32Array([0, 0]),
    };

    agg.submitUpdate({ weights: clientA, interactionCount: 75 });
    agg.submitUpdate({ weights: clientB, interactionCount: 25 });

    const result = agg.aggregate();

    // W_new = I + (75/100)*0.1 + (25/100)*0.5 = 1 + 0.075 + 0.125 = 1.2
    expect(result.matrix[0]).toBeCloseTo(1.2, 5);
  });

  it("reset clears updates and optionally updates base", () => {
    const base = makeIdentityWeights();
    const agg = new FederatedAggregator(base, DIM);

    agg.submitUpdate({
      weights: {
        matrix: new Float32Array([2, 0, 0, 2]),
        bias: new Float32Array([1, 1]),
      },
      interactionCount: 10,
    });

    expect(agg.clientCount).toBe(1);

    const newBase: IntentWeights = {
      matrix: new Float32Array([1.5, 0, 0, 1.5]),
      bias: new Float32Array([0.5, 0.5]),
    };
    agg.reset(newBase);

    expect(agg.clientCount).toBe(0);
  });

  it("throws when no updates submitted", () => {
    const agg = new FederatedAggregator(makeIdentityWeights(), DIM);
    expect(() => agg.aggregate()).toThrow("No client updates submitted");
  });

  it("throws when interactionCount is 0", () => {
    const agg = new FederatedAggregator(makeIdentityWeights(), DIM);
    expect(() =>
      agg.submitUpdate({
        weights: makeIdentityWeights(),
        interactionCount: 0,
      }),
    ).toThrow("interactionCount must be greater than 0");
  });

  it("bias aggregation works correctly", () => {
    const base = makeIdentityWeights();
    const agg = new FederatedAggregator(base, DIM);

    agg.submitUpdate({
      weights: {
        matrix: new Float32Array([1, 0, 0, 1]),
        bias: new Float32Array([1.0, 2.0]),
      },
      interactionCount: 50,
    });
    agg.submitUpdate({
      weights: {
        matrix: new Float32Array([1, 0, 0, 1]),
        bias: new Float32Array([3.0, 4.0]),
      },
      interactionCount: 50,
    });

    const result = agg.aggregate();
    // bias = [0,0] + 0.5*([1,2]-[0,0]) + 0.5*([3,4]-[0,0]) = [2.0, 3.0]
    expect(result.bias[0]).toBeCloseTo(2.0, 5);
    expect(result.bias[1]).toBeCloseTo(3.0, 5);
  });
});
