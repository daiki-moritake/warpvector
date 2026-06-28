import { expect, test, describe, beforeAll } from "bun:test";
import { MoeAdapter } from "../src/adapters/MoeAdapter";
import { IntentAdapter } from "@warpvector/core";

describe("MoeAdapter", () => {
  test("routes to the correct expert based on cosine similarity", async () => {
    // 2つのエキスパート（IntentAdapter）を作成
    const expert1 = new IntentAdapter({
      test: {
        matrix: [
          [2, 0],
          [0, 2],
        ],
        bias: [0, 0],
      },
    }); // 2倍にするエキスパート

    const expert2 = new IntentAdapter({
      test: {
        matrix: [
          [-1, 0],
          [0, -1],
        ],
        bias: [0, 0],
      },
    }); // 符号を反転させるエキスパート

    const moe = new MoeAdapter({
      experts: [
        {
          id: "expert1",
          adapter: expert1,
          centroid: [1, 0], // x軸方向の入力は expert1 へ
        },
        {
          id: "expert2",
          adapter: expert2,
          centroid: [0, 1], // y軸方向の入力は expert2 へ
        },
      ],
    });

    await moe.init();

    // x軸に近いベクトル -> expert1 が選ばれるはず（[2, 0]になる）
    const out1 = moe.tune([1, 0.1], "test");
    expect(out1[0]).toBeCloseTo(2, 5);
    expect(out1[1]).toBeCloseTo(0.2, 5);

    // y軸に近いベクトル -> expert2 が選ばれるはず（[-0.1, -1]になる）
    const out2 = moe.tune([0.1, 1], "test");
    expect(out2[0]).toBeCloseTo(-0.1, 5);
    expect(out2[1]).toBeCloseTo(-1, 5);
  });

  test("uses custom routing strategy", async () => {
    const expert1 = new IntentAdapter({
      test: { matrix: [[1]], bias: [1] }, // +1 する
    });
    const expert2 = new IntentAdapter({
      test: { matrix: [[1]], bias: [-1] }, // -1 する
    });

    const moe = new MoeAdapter({
      routingStrategy: "custom",
      customRouter: (vec: Float32Array) => {
        // 第一要素が正なら expert1、負なら expert2
        return vec[0] >= 0 ? "expert1" : "expert2";
      },
      experts: [
        { id: "expert1", adapter: expert1 },
        { id: "expert2", adapter: expert2 },
      ],
    });

    const out1 = moe.tune([5], "test"); // 5 >= 0 -> expert1 -> 5 + 1 = 6
    expect(out1[0]).toBeCloseTo(6, 5);

    const out2 = moe.tune([-5], "test"); // -5 < 0 -> expert2 -> -5 - 1 = -6
    expect(out2[0]).toBeCloseTo(-6, 5);
  });

  test("exportState and importState work correctly", async () => {
    // 状態の復元ができるかテストするために、AdapterRegistry が必要だが、
    // ml パッケージのテスト環境では @warpvector/core の AdapterRegistry が利用可能。
    // IntentAdapter は core パッケージで自動登録されている。

    const expert1 = new IntentAdapter({
      test: { matrix: [[2]], bias: [0] },
    });

    const moe = new MoeAdapter({
      experts: [
        {
          id: "expert1",
          adapter: expert1,
          centroid: [1],
        },
      ],
    });

    const state = moe.exportState();
    expect(typeof state).toBe("string");

    const parsed = JSON.parse(state as string);
    expect(parsed.type).toBe("MoeAdapter");
    expect(parsed.experts[0].id).toBe("expert1");

    const restoredMoe = MoeAdapter.importState(state);

    // 復元後も同じように動作するか
    const out = restoredMoe.tune([5], "test");
    expect(out[0]).toBeCloseTo(10, 5);
  });

  test("tuneBatch works correctly", async () => {
    const expert1 = new IntentAdapter({
      test: { matrix: [[2]], bias: [0] },
    });
    const moe = new MoeAdapter({
      experts: [{ id: "e1", adapter: expert1 }],
    });

    const out = moe.tuneBatch([[1], [2], [3]], "test");
    expect(out.length).toBe(3);
    expect(out[0][0]).toBe(2);
    expect(out[1][0]).toBe(4);
    expect(out[2][0]).toBe(6);
  });
});
