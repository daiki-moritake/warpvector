import { expect, test, describe } from "bun:test";
import { TripletTrainer } from "../src/trainers/TripletTrainer";
import { IntentAdapter, IntentWeights, type TripletExample, cosineSimilarity } from "@warpvector/core";

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

  test("resets Adam state completely on multiple batch training runs", async () => {
    const trainer = new TripletTrainer(3);
    trainer.addExample({
      anchor: [1, 0, 0],
      positive: [0.8, 0.2, 0.0],
      negative: [0.9, 0.0, 0.1],
    });

    const weights1 = await trainer.train({ epochs: 10 });
    const weights2 = await trainer.train({ epochs: 10 });

    expect(Array.from(weights1.matrix as Float32Array)).toEqual(
      Array.from(weights2.matrix as Float32Array),
    );
    expect(Array.from(weights1.bias as Float32Array)).toEqual(
      Array.from(weights2.bias as Float32Array),
    );
  });

  test("rejects invalid hyperparameters in train and updateOnline", async () => {
    const trainer = new TripletTrainer(3);
    trainer.addExample({
      anchor: [1, 0, 0],
      positive: [0.8, 0.2, 0.0],
      negative: [0.9, 0.0, 0.1],
    });

    await expect(trainer.train({ learningRate: -0.1 })).rejects.toThrow(
      "learningRate",
    );
    await expect(trainer.train({ regularization: -0.01 })).rejects.toThrow(
      "regularization",
    );
    await expect(trainer.train({ epochs: 1.5 })).rejects.toThrow("epochs");
    await expect(trainer.train({ margin: -1.0 } as any)).rejects.toThrow(
      "margin",
    );

    const dummyWeights = {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      bias: [0, 0, 0],
    };
    const dummyExample = {
      anchor: [1, 0, 0],
      positive: [0.8, 0.2, 0.0],
      negative: [0.9, 0.0, 0.1],
    };

    await expect(
      trainer.updateOnline(dummyWeights, dummyExample, { margin: -0.5 }),
    ).rejects.toThrow("margin");
    await expect(
      trainer.updateOnline(dummyWeights, dummyExample, { learningRate: 0 }),
    ).rejects.toThrow("learningRate");
  });
});
