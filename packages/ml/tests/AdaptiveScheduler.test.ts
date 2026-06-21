import { describe, it, expect } from "bun:test";
import { TripletTrainer } from "../src/trainers/TripletTrainer";
import { AdaptiveScheduler } from "../src/feedback/AdaptiveScheduler";
import type { TripletExample } from "../src/trainers/TripletTrainer";
import type { IntentWeights } from "@warpvector/core";

const DIM = 4;

function makeIdentityWeights(): IntentWeights {
  const matrix = new Float32Array(DIM * DIM);
  for (let i = 0; i < DIM; i++) matrix[i * DIM + i] = 1.0;
  return { matrix, bias: new Float32Array(DIM) };
}

function makeExample(seed: number): TripletExample {
  const anchor = new Float32Array(DIM);
  const positive = new Float32Array(DIM);
  const negative = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) {
    anchor[i] = seed * 0.1 + i * 0.01;
    positive[i] = anchor[i] + 0.05;
    negative[i] = anchor[i] - 0.05;
  }
  return { anchor, positive, negative };
}

describe("AdaptiveScheduler", () => {
  it("returns null when buffer is below batchSize", async () => {
    const trainer = new TripletTrainer(DIM);
    const scheduler = new AdaptiveScheduler(trainer, { batchSize: 5 });
    const weights = makeIdentityWeights();

    const result = await scheduler.addFeedback(weights, [makeExample(1)]);
    expect(result).toBeNull();
    expect(scheduler.bufferedCount).toBe(1);
  });

  it("trains when batchSize is reached and returns updated weights", async () => {
    const trainer = new TripletTrainer(DIM);
    const scheduler = new AdaptiveScheduler(trainer, { batchSize: 3 });
    const weights = makeIdentityWeights();

    // 2件 → まだ学習しない
    await scheduler.addFeedback(weights, [makeExample(1), makeExample(2)]);
    expect(scheduler.totalSteps).toBe(0);

    // 3件目 → batchSize 達成 → 学習実行
    const updated = await scheduler.addFeedback(weights, [makeExample(3)]);
    expect(updated).not.toBeNull();
    expect(scheduler.totalSteps).toBe(3);
  });

  it("learning rate decays with totalSteps", () => {
    const trainer = new TripletTrainer(DIM);
    const scheduler = new AdaptiveScheduler(trainer, {
      initialLearningRate: 0.01,
      decayRate: 0.1,
      minLearningRate: 0.001,
    });

    const lr0 = scheduler.currentLearningRate;
    expect(lr0).toBe(0.01);

    // 手動で totalSteps を増やすため、importState を使用
    const state = JSON.parse(scheduler.exportState());
    state.totalSteps = 100;
    const restored = AdaptiveScheduler.importState(
      trainer,
      JSON.stringify(state),
    );

    // lr(100) = max(0.001, 0.01 / (1 + 0.1 * 100)) = max(0.001, 0.01/11) ≈ 0.000909
    // → min に丸められて 0.001
    expect(restored.currentLearningRate).toBeCloseTo(0.001, 3);
  });

  it("flushAndTrain processes remaining buffer", async () => {
    const trainer = new TripletTrainer(DIM);
    const scheduler = new AdaptiveScheduler(trainer, { batchSize: 10 });
    const weights = makeIdentityWeights();

    await scheduler.addFeedback(weights, [makeExample(1), makeExample(2)]);
    expect(scheduler.bufferedCount).toBe(2);

    const updated = await scheduler.flushAndTrain(weights);
    expect(updated).not.toBeNull();
    expect(scheduler.bufferedCount).toBe(0);
    expect(scheduler.totalSteps).toBe(2);
  });

  it("exportState and importState preserve totalSteps", () => {
    const trainer = new TripletTrainer(DIM);
    const scheduler = new AdaptiveScheduler(trainer, {
      initialLearningRate: 0.05,
      batchSize: 10,
    });

    const json = scheduler.exportState();
    const restored = AdaptiveScheduler.importState(trainer, json);

    expect(restored.totalSteps).toBe(0);
    expect(restored.currentLearningRate).toBe(0.05);
  });
});
