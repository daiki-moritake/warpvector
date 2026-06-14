import { expect, test, describe } from "bun:test";
import { MigrationTrainer } from "../src/migration";
import { ProjectionAdapter } from "../src/ProjectionAdapter";
import { cosineSimilarity } from "../src/utils";

describe("MigrationTrainer", () => {
  test("trains a mapping from 4D to 2D", async () => {
    // 古いモデル(4D) から 新しいモデル(2D) へのマイグレーションを学習する
    const trainer = new MigrationTrainer(4, 2);

    trainer.addExample({
      source: [1.0, 0.0, 0.0, 0.0],
      target: [0.5, 0.5],
    });
    trainer.addExample({
      source: [0.0, 1.0, 0.0, 0.0],
      target: [-0.5, 0.5],
    });
    trainer.addExample({
      source: [0.0, 0.0, 1.0, 0.0],
      target: [0.5, -0.5],
    });
    trainer.addExample({
      source: [0.0, 0.0, 0.0, 1.0],
      target: [-0.5, -0.5],
    });

    const learnedWeights = await trainer.train({
      learningRate: 0.1,
      epochs: 300,
      regularization: 0.001,
    });

    // ProjectionAdapter に学習した重みをセット
    const adapter = new ProjectionAdapter(4, 2);
    adapter.addProjection("model_v1_to_v2", learnedWeights);

    // テスト
    const res1 = adapter.tune([1.0, 0.0, 0.0, 0.0], "model_v1_to_v2");
    expect(cosineSimilarity(res1, [0.5, 0.5])).toBeGreaterThan(0.95);

    const res2 = adapter.tune([0.0, 1.0, 0.0, 0.0], "model_v1_to_v2");
    expect(cosineSimilarity(res2, [-0.5, 0.5])).toBeGreaterThan(0.95);

    const res3 = adapter.tune([0.0, 0.0, 1.0, 0.0], "model_v1_to_v2");
    expect(cosineSimilarity(res3, [0.5, -0.5])).toBeGreaterThan(0.95);

    const res4 = adapter.tune([0.0, 0.0, 0.0, 1.0], "model_v1_to_v2");
    expect(cosineSimilarity(res4, [-0.5, -0.5])).toBeGreaterThan(0.95);
  });

  test("trains a mapping from 2D to 3D (Expansion)", async () => {
    // 逆のパターン: 次元拡張
    const trainer = new MigrationTrainer(2, 3);

    trainer.addExample({
      source: [1.0, 0.0],
      target: [0.5, 0.5, 0.0],
    });
    trainer.addExample({
      source: [0.0, 1.0],
      target: [0.0, 0.5, 0.5],
    });

    const learnedWeights = await trainer.train({
      learningRate: 0.1,
      epochs: 300,
    });

    const adapter = new ProjectionAdapter(2, 3);
    adapter.addProjection("expand", learnedWeights);

    const res1 = adapter.tune([1.0, 0.0], "expand");
    expect(cosineSimilarity(res1, [0.5, 0.5, 0.0])).toBeGreaterThan(0.9);
  });
});
