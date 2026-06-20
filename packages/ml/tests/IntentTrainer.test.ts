import { expect, test, describe } from "bun:test";
import { IntentTrainer, TrainingExample } from "../src/IntentTrainer";
import { IntentAdapter } from "@warpvector/core";
import { cosineSimilarity } from "@warpvector/core";

describe("IntentTrainer", () => {
  test("trains W and b to map input to target (batch training)", async () => {
    const dimension = 3;
    const trainer = new IntentTrainer(dimension);

    // 学習データ: 3つの入力ベクトルと、理想とするターゲットベクトル
    // 例えば [1, 0, 0] を [0.5, 0.5, 0.5] に近づけるような変換を学習させる
    trainer.addExample({
      input: [1.0, 0.0, 0.0],
      target: [0.5, 0.5, 0.5],
    });
    trainer.addExample({
      input: [0.0, 1.0, 0.0],
      target: [0.1, 0.9, 0.1],
    });
    trainer.addExample({
      input: [0.0, 0.0, 1.0],
      target: [0.1, 0.1, 0.9],
    });

    // 十分なエポック数で学習 (少し高めの学習率で早く収束させる)
    const learnedWeights = await trainer.train({
      learningRate: 0.05,
      epochs: 300,
      regularization: 0.001,
    });

    // 学習された IntentWeights を使って変換を実行
    const adapter = new IntentAdapter({ learnedIntent: learnedWeights });

    // 学習に使ったベクトルを入力してみる
    const result1 = adapter.tune([1.0, 0.0, 0.0], "learnedIntent");

    // 目標値 [0.5, 0.5, 0.5] との類似度が非常に高くなっているはず
    const sim1 = cosineSimilarity(result1, [0.5, 0.5, 0.5]);
    expect(sim1).toBeGreaterThan(0.99);

    const result2 = adapter.tune([0.0, 1.0, 0.0], "learnedIntent");
    const sim2 = cosineSimilarity(result2, [0.1, 0.9, 0.1]);
    expect(sim2).toBeGreaterThan(0.99);
  });

  test("trains W and b with autoTune enabled", async () => {
    const dimension = 3;
    const trainer = new IntentTrainer(dimension);

    trainer.addExample({
      input: [1.0, 0.0, 0.0],
      target: [0.5, 0.5, 0.5],
    });
    trainer.addExample({
      input: [0.0, 1.0, 0.0],
      target: [0.1, 0.9, 0.1],
    });

    const learnedWeights = await trainer.train({
      epochs: 300,
      autoTune: true, // 自動チューニングを有効化
    });

    const adapter = new IntentAdapter({ learnedIntent: learnedWeights });
    const result1 = adapter.tune([1.0, 0.0, 0.0], "learnedIntent");
    const sim1 = cosineSimilarity(result1, [0.5, 0.5, 0.5]);
    expect(sim1).toBeGreaterThan(0.9); // チューニングにより正しく学習されること
  });

  test("updateOnline adapts weights iteratively", async () => {
    const dimension = 2;
    const trainer = new IntentTrainer(dimension);

    // 初期の重み (単位行列とゼロバイアス)
    const initialWeights = {
      matrix: [
        [1.0, 0.0],
        [0.0, 1.0],
      ],
      bias: [0.0, 0.0],
    };

    const inputVector = [1.0, 0.0];
    const targetVector = [0.0, 1.0]; // ユーザーがクリックした（求めていた）理想のベクトル

    const example: TrainingExample = {
      input: inputVector,
      target: targetVector,
    };

    // 1回のオンラインアップデート (強めの学習率で変化を確認)
    const updatedWeights1 = await trainer.updateOnline(
      initialWeights,
      example,
      {
        learningRate: 0.5,
        regularization: 0.0,
      },
    );

    const adapter1 = new IntentAdapter({ intent: updatedWeights1 });
    const res1 = adapter1.tune(inputVector, "intent");

    // 初期状態では res = [1.0, 0.0] だが、1回のアップデートで target=[0.0, 1.0] に少し近づくはず
    const initialSim = cosineSimilarity([1.0, 0.0], targetVector); // 0.0
    const updatedSim1 = cosineSimilarity(res1, targetVector);
    expect(updatedSim1).toBeGreaterThan(initialSim);

    // さらに何回かアップデートすると、どんどん近づいていく
    let currentWeights = updatedWeights1;
    for (let i = 0; i < 5; i++) {
      currentWeights = await trainer.updateOnline(
        currentWeights,
        {
          input: inputVector,
          target: targetVector,
        },
        {
          learningRate: 0.5,
          regularization: 0.0,
        },
      );
    }

    const adapter2 = new IntentAdapter({ intent: currentWeights });
    const res2 = adapter2.tune(inputVector, "intent");
    const updatedSim2 = cosineSimilarity(res2, targetVector);

    // より目標に近づいている（または既に完全に一致している）はず
    expect(updatedSim2).toBeGreaterThan(0.9);
  });
});
