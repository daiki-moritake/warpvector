import { expect, test, describe } from "bun:test";
import { applyActivationToVector, softmax } from "@warpvector/core";
import { IntentAdapter } from "@warpvector/core";

describe("Math Extensions (数学モデルの拡張)", () => {
  test("Softmax function (Softmax関数の計算)", () => {
    // 3つの値に対するSoftmaxをテスト
    const values = [1, 2, 3];
    const result = softmax(values);
    expect(result.length).toBe(3);

    // 計算結果が数学的な期待値と一致するか確認
    expect(Math.abs(result[0] - 0.09003057)).toBeLessThan(0.0001);
    expect(Math.abs(result[1] - 0.24472847)).toBeLessThan(0.0001);
    expect(Math.abs(result[2] - 0.66524096)).toBeLessThan(0.0001);

    // 確率の合計が1になることを確認
    expect(result[0] + result[1] + result[2]).toBeCloseTo(1);
  });

  test("Activations (非線形活性化関数: ReLU, Sigmoid, Tanh)", () => {
    // ReLU: 負の値を0にする
    const vecReLU = new Float32Array([-1, 0, 1]);
    applyActivationToVector(vecReLU, "relu");
    expect(vecReLU[0]).toBe(0);
    expect(vecReLU[1]).toBe(0);
    expect(vecReLU[2]).toBe(1);

    // Sigmoid: 0を入力すると0.5を返す
    const vecSigmoid = new Float32Array([0]);
    applyActivationToVector(vecSigmoid, "sigmoid");
    expect(vecSigmoid[0]).toBe(0.5);

    // Tanh: 0を入力すると0を返す
    const vecTanh = new Float32Array([0]);
    applyActivationToVector(vecTanh, "tanh");
    expect(vecTanh[0]).toBe(0);
  });

  test("tuneAutoBlended (自己アテンション型動的ブレンド)", () => {
    // 2つのインテントを定義し、それぞれに routingVector を設定
    const adapter = new IntentAdapter({
      intentA: {
        matrix: [
          [1, 0],
          [0, 1],
        ],
        bias: [1, 1],
        routingVector: [1, 0], // x軸方向に近いベクトル
      },
      intentB: {
        matrix: [
          [1, 0],
          [0, 1],
        ],
        bias: [-1, -1],
        routingVector: [0, 1], // y軸方向に近いベクトル
      },
    });

    // x軸方向に近いベースベクトルを入力
    const baseVector = [1, 0.1];
    const blended = adapter.tuneAutoBlended(baseVector);

    // intentA のウェイトが高くなるため、結果は intentA のバイアス [1, 1] に近づくはず
    expect(blended[0]).toBeGreaterThan(1);
    expect(blended[1]).toBeGreaterThan(0);
  });
});
