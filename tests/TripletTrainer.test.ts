import { expect, test, describe } from "bun:test";
import { TripletTrainer, TripletExample } from "../src/TripletTrainer";
import { IntentAdapter, IntentWeights } from "../src/IntentAdapter";
import { cosineSimilarity } from "../src/utils";

describe("TripletTrainer", () => {
  test("trains W and b to map anchor closer to positive and further from negative", async () => {
    const dimension = 3;
    const trainer = new TripletTrainer(dimension);

    // Anchor: [1, 0, 0] (検索クエリ)
    // Positive: [0.8, 0.2, 0.0] (正解ドキュメント、本来近いべき)
    // Negative: [0.9, 0.0, 0.1] (不正解ドキュメント、最初は近いが遠ざけるべき)
    const anchor = [1.0, 0.0, 0.0];
    const positive = [0.8, 0.2, 0.0];
    const negative = [0.9, 0.0, 0.1];

    // 学習前の状態を確認
    const initialSimPos = cosineSimilarity(anchor, positive);
    const initialSimNeg = cosineSimilarity(anchor, negative);

    // [1, 0, 0] と [0.9, 0.0, 0.1] はかなり近い
    expect(initialSimNeg).toBeGreaterThan(0.9);

    // 初期の重み (単位行列とゼロバイアス)
    let currentWeights: IntentWeights = {
      matrix: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
      ],
      bias: [0.0, 0.0, 0.0],
    };

    const example: TripletExample = {
      anchor,
      positive,
      negative,
    };

    // 100エポック学習 (オンライン更新をシミュレート)
    for (let epoch = 0; epoch < 100; epoch++) {
      currentWeights = await trainer.updateOnline(currentWeights, example, {
        learningRate: 0.05,
        margin: 0.1,
        regularization: 0.001,
      });
    }

    const adapter = new IntentAdapter({ learnedIntent: currentWeights });
    const warpedAnchor = adapter.tune(anchor, "learnedIntent");

    // 学習後の状態を確認
    const finalSimPos = cosineSimilarity(warpedAnchor, positive);
    const finalSimNeg = cosineSimilarity(warpedAnchor, negative);

    // 正解ベクトルには近づき、不正解ベクトルからは遠ざかっていること
    // Margin Loss によって空間が歪められる
    const initialDiff = initialSimPos - initialSimNeg;
    const finalDiff = finalSimPos - finalSimNeg;

    expect(finalDiff).toBeGreaterThan(initialDiff);
  });

  test("throws error if dimension mismatch", async () => {
    const trainer = new TripletTrainer(3);

    // updateOnlineは非同期メソッドなので、rejectsをキャッチする
    await expect(
      trainer.updateOnline(
        {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
          bias: [0, 0, 0],
        },
        {
          anchor: [1, 0], // 次元が違う
          positive: [0, 1, 0],
          negative: [0, 0, 1],
        },
      ),
    ).rejects.toThrow();
  });
});
