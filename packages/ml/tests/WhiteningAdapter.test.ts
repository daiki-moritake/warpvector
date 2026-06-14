import { expect, test, describe } from "bun:test";
import { WhiteningAdapter } from "../src/WhiteningAdapter";
import { cosineSimilarity, normalize } from "@warpvector/core";

describe("WhiteningAdapter", () => {
  test("Online PCA removes the primary bias direction and reduces anisotropy", () => {
    const dim = 10;
    const adapter = new WhiteningAdapter(dim, {
      learningRate: 0.05,
      numComponents: 1,
    });

    // 偏りのあるベクトルを生成する (Anisotropic distribution)
    // 特定の次元 (例えばインデックス0, 1) に巨大な分散を持たせる
    const generateBiasedVector = () => {
      const v = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        v[i] = Math.random() - 0.5;
      }
      // 0番目の次元だけ分散を10倍にする
      v[0] = (Math.random() - 0.5) * 10.0;
      return v;
    };

    const numSamples = 500;
    const testVectors: Float32Array[] = [];

    // 1. 学習 (ストリーミング入力)
    for (let i = 0; i < numSamples; i++) {
      const v = generateBiasedVector();
      testVectors.push(v);
      adapter.update(v);
    }

    // 2. 学習後の主成分(PC1)が分散の大きい方向 ([1, 0, 0, ...]) に近いか確認
    const pc1 = adapter.components[0];

    // 0番目の次元が極めて大きいはず (絶対値がほぼ1)
    expect(Math.abs(pc1[0])).toBeGreaterThan(0.9);

    // 3. 学習前と学習後のコサイン類似度を比較
    // ランダムに選んだ2つのベクトル間の類似度
    let originalSimSum = 0;
    let whitenedSimSum = 0;
    const pairs = 100;

    for (let i = 0; i < pairs; i++) {
      const v1 = testVectors[Math.floor(Math.random() * numSamples)];
      const v2 = testVectors[Math.floor(Math.random() * numSamples)];

      // 元のベクトルは0次元目の分散が支配的なので、類似度が高くなりやすい (相関しやすい)
      originalSimSum += Math.abs(cosineSimilarity(v1, v2));

      // Whitening後
      const w1 = adapter.tune(v1);
      const w2 = adapter.tune(v2);
      whitenedSimSum += Math.abs(cosineSimilarity(w1, w2));
    }

    const avgOriginalSim = originalSimSum / pairs;
    const avgWhitenedSim = whitenedSimSum / pairs;

    // Whiteningによって支配的な成分が除去され、無相関になるため類似度の絶対値平均は低下する
    expect(avgWhitenedSim).toBeLessThan(avgOriginalSim);
  });

  test("throws error if dimension mismatch", () => {
    const adapter = new WhiteningAdapter(5);
    expect(() => adapter.update([1, 2, 3, 4])).toThrow();
    expect(() => adapter.tune([1, 2, 3, 4, 5, 6])).toThrow();
  });
});
