import { expect, test, describe } from "bun:test";
import { InfoNCETrainer } from "../src/InfoNCETrainer";
import { IntentAdapter, IntentWeights } from "../src/IntentAdapter";
import { cosineSimilarity } from "../src/utils";

describe("InfoNCETrainer", () => {
  test("trains W and b to map anchor closer to positive and further from multiple negatives", async () => {
    const dimension = 3;
    const trainer = new InfoNCETrainer(dimension);

    // Anchor: 検索クエリ
    const anchor = [1.0, 0.0, 0.0];
    // Positive: クリックされた正解ドキュメント
    const positive = [0.8, 0.2, 0.0];
    // Negatives: スルーされた複数の不正解ドキュメント
    const negatives = [
      [0.9, 0.0, 0.1], // Negative 1
      [0.85, 0.1, 0.0], // Negative 2
      [1.0, 0.0, 0.0], // Negative 3 (Anchorと全く同じベクトルだが遠ざけたい)
    ];

    // 初期の重み (単位行列とゼロバイアス)
    let currentWeights: IntentWeights = {
      matrix: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
      ],
      bias: [0.0, 0.0, 0.0],
    };

    // 100エポック学習
    for (let epoch = 0; epoch < 100; epoch++) {
      currentWeights = await trainer.updateOnline(
        currentWeights,
        anchor,
        positive,
        negatives,
        {
          learningRate: 0.05,
          temperature: 0.1,
          regularization: 0.001
        }
      );
    }

    const adapter = new IntentAdapter({ learnedIntent: currentWeights });
    const warpedAnchor = adapter.tune(anchor, "learnedIntent");

    // 学習後の状態を確認
    const finalSimPos = cosineSimilarity(warpedAnchor, positive);
    
    // 正解ベクトルへの類似度が、すべてのNegativeへの類似度よりも高くなっていることを確認
    for (const neg of negatives) {
      const simNeg = cosineSimilarity(warpedAnchor, neg);
      expect(finalSimPos).toBeGreaterThan(simNeg);
    }
  });

  test("throws error if dimension mismatch", async () => {
    const trainer = new InfoNCETrainer(3);
    
    await expect(
      trainer.updateOnline(
        {
          matrix: [[1,0,0],[0,1,0],[0,0,1]],
          bias: [0,0,0]
        },
        [1, 0], // 次元が違う
        [0, 1, 0],
        [[0, 0, 1]]
      )
    ).rejects.toThrow();

    await expect(
      trainer.updateOnline(
        {
          matrix: [[1,0,0],[0,1,0],[0,0,1]],
          bias: [0,0,0]
        },
        [1, 0, 0],
        [0, 1, 0],
        [[0, 0, 1], [1, 0]] // negativesの中に次元が違うものが混ざっている
      )
    ).rejects.toThrow();
  });
});
