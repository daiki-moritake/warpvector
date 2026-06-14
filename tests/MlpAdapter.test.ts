import { expect, test, describe, beforeAll } from "bun:test";
import { MlpAdapter, MlpLayer } from "../src/MlpAdapter";
import { initWasm } from "../src/wasm/wasm-loader";

describe("MlpAdapter", () => {
  beforeAll(async () => {
    await initWasm();
  });

  test("runs a multi-layer network with ReLU and Sigmoid correctly", async () => {
    // 3次元 -> 4次元 (ReLU) -> 2次元 (Sigmoid)
    const layers: MlpLayer[] = [
      {
        matrix: [
          [1, 0, -1],
          [0, 1, 0],
          [-1, 0, 1],
          [0.5, 0.5, 0.5]
        ],
        bias: [0, -1, 0, 0],
        activation: "relu"
      },
      {
        matrix: [
          [1, 1, 1, 1],
          [1, -1, 0, 0]
        ],
        bias: [-1, 0],
        activation: "sigmoid"
      }
    ];

    const adapter = new MlpAdapter(layers);
    await adapter.init();

    const input = [1, 2, 3];
    const outWasm = adapter.tune(input);

    // TypeScript側での純粋な計算
    // Layer 1
    // [1, 0, -1] * [1,2,3] + 0 = -2 => ReLU => 0
    // [0, 1, 0] * [1,2,3] - 1 = 1 => ReLU => 1
    // [-1, 0, 1] * [1,2,3] + 0 = 2 => ReLU => 2
    // [0.5, 0.5, 0.5] * [1,2,3] + 0 = 3 => ReLU => 3
    const h1 = [0, 1, 2, 3];

    // Layer 2
    // [1, 1, 1, 1] * [0, 1, 2, 3] - 1 = 6 - 1 = 5 => Sigmoid(5)
    // [1, -1, 0, 0] * [0, 1, 2, 3] + 0 = -1 => Sigmoid(-1)
    
    const expectedOut = [
      1.0 / (1.0 + Math.exp(-5)),
      1.0 / (1.0 + Math.exp(1))
    ];

    expect(outWasm.length).toBe(2);
    expect(outWasm[0]).toBeCloseTo(expectedOut[0], 5);
    expect(outWasm[1]).toBeCloseTo(expectedOut[1], 5);
  });

  test("runs a single linear layer correctly", async () => {
    const layers: MlpLayer[] = [
      {
        matrix: [
          [2, 0],
          [0, 3]
        ],
        bias: [1, -1],
        activation: "linear"
      }
    ];

    const adapter = new MlpAdapter(layers);
    await adapter.init();

    const input = [5, 4];
    const outWasm = adapter.tune(input);

    expect(outWasm[0]).toBeCloseTo(11); // 2*5 + 1
    expect(outWasm[1]).toBeCloseTo(11); // 3*4 - 1
  });

  test("throws error when passing incorrect input dimension", async () => {
    const layers: MlpLayer[] = [
      {
        matrix: [
          [1, 0],
          [0, 1]
        ],
        bias: [0, 0],
        activation: "linear"
      }
    ];

    const adapter = new MlpAdapter(layers);
    await adapter.init();

    expect(() => adapter.tune([1])).toThrow();
    expect(() => adapter.tune([1, 2, 3])).toThrow();
  });
});
