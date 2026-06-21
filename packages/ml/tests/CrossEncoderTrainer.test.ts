import { expect, test, describe } from "bun:test";
import { CrossEncoderTrainer } from "../src/trainers/CrossEncoderTrainer";

describe("CrossEncoderTrainer", () => {
  test("trains a model to output expected scores based on query and document", async () => {
    // クエリ次元 2, ドキュメント次元 2
    const trainer = new CrossEncoderTrainer(2, 2);

    // 単純なルール：
    // クエリとドキュメントが同じ方向ならスコア1.0
    // 逆方向ならスコア0.0
    trainer.addExample({
      query: [1, 0],
      document: [1, 0],
      score: 1.0
    });
    trainer.addExample({
      query: [0, 1],
      document: [0, 1],
      score: 1.0
    });
    trainer.addExample({
      query: [1, 0],
      document: [-1, 0],
      score: 0.0
    });
    trainer.addExample({
      query: [0, 1],
      document: [0, -1],
      score: 0.0
    });

    const weights = await trainer.train({
      epochs: 500, // 十分な学習回数
      learningRate: 0.05,
      regularization: 0.0
    });

    expect(weights.matrix).toBeDefined();
    expect(weights.matrix.length).toBe(4); // (queryDim + docDim) * targetDim = (2 + 2) * 1 = 4
    expect(weights.bias).toBeDefined();
    expect(weights.bias.length).toBe(1);

    // テスト：学習した重みを使って予測
    const predict = (q: number[], d: number[]) => {
      const input = [...q, ...d];
      let score = weights.bias[0];
      for (let i = 0; i < input.length; i++) {
        score += input[i] * weights.matrix[i];
      }
      return score;
    };

    // 学習データに対する予測精度を確認
    const score1 = predict([1, 0], [1, 0]);
    expect(score1).toBeGreaterThan(0.8);

    const score2 = predict([1, 0], [-1, 0]);
    expect(score2).toBeLessThan(0.2);
  });

  test("autoTune finds optimal learning rate", async () => {
    const trainer = new CrossEncoderTrainer(2); // queryDim=2, docDim=2
    trainer.addExample({ query: [1, 1], document: [1, 1], score: 1.0 });
    trainer.addExample({ query: [1, 1], document: [-1, -1], score: 0.0 });

    const weights = await trainer.train({ epochs: 10, autoTune: true });
    expect(weights.matrix.length).toBe(4);
  });
});
