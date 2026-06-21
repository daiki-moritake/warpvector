import { expect, test, describe, beforeAll } from "bun:test";
import { WarpPipeline, IntentAdapter } from "@warpvector/core";
import { computeCosineSimilarity, getPositiveRank, calculateMRR, calculateRecall } from "../src/automl/metrics";
import { PipelineAutoTuner } from "../src/automl/PipelineAutoTuner";
import { SearchExample } from "../src/automl/metrics";

import { InputVector } from "@warpvector/core";

describe("AutoML Metrics", () => {
  test("computeCosineSimilarity", () => {
    const a = [1, 0];
    const b = [0, 1];
    const c = [1, 0];
    expect(computeCosineSimilarity(a, b)).toBeCloseTo(0);
    expect(computeCosineSimilarity(a, c)).toBeCloseTo(1);
  });

  test("getPositiveRank", () => {
    const query = [1, 0];
    const positive = [0.9, 0.1]; // 高い類似度
    const neg1 = [0, 1]; // 低い類似度
    const neg2 = [0.95, 0.05]; // queryにより近い不正解
    
    // posのみの場合、ランクは1
    expect(getPositiveRank(query, positive, [])).toBe(1);
    // neg1よりposの方が似ているのでランクは1
    expect(getPositiveRank(query, positive, [neg1])).toBe(1);
    // neg2の方がqueryに近いので、posのランクは2に下がる
    expect(getPositiveRank(query, positive, [neg1, neg2])).toBe(2);
  });

  test("calculateMRR and Recall", () => {
    const dataset: SearchExample<InputVector>[] = [
      {
        query: [1, 0],
        positive: [0.9, 0.1],
        negatives: [[0, 1]]
      }, // ランク1 -> RR 1.0
      {
        query: [0, 1],
        positive: [0.1, 0.9],
        negatives: [[0, 0.95], [1, 0]]
      } // ランク2 -> RR 0.5
    ];

    const mrr = calculateMRR(dataset);
    expect(mrr).toBeCloseTo((1.0 + 0.5) / 2);

    expect(calculateRecall(dataset, 1)).toBe(0.5); // 半分が1位
    expect(calculateRecall(dataset, 2)).toBe(1.0); // 全部が2位以内
  });
});

describe("PipelineAutoTuner", () => {
  test("tunes parameters with Grid Search", async () => {
    // 非常にシンプルな検証データ
    // query と positive が一致しているが、スケーリングが異なるなど
    const dataset: SearchExample<InputVector>[] = [
      {
        query: [1, 0.1],
        positive: [1, 0],
        negatives: [[-1, 0]]
      },
      {
        query: [0.1, 1],
        positive: [0, 1],
        negatives: [[0, -1]]
      }
    ];

    const tuner = new PipelineAutoTuner(dataset);

    const result = await tuner.tuneGrid({
      searchSpace: {
        multiplier: [1, 5, 10], // 探索するパラメータ
        bias: [0, 1]
      },
      pipelineBuilder: (params) => {
        const pipeline = new WarpPipeline(2);
        // パラメータを使ってアダプタを構築
        const adapter = new IntentAdapter({
          default: {
            matrix: [
              [params.multiplier, 0],
              [0, params.multiplier]
            ],
            bias: [params.bias, params.bias]
          }
        });
        pipeline.addStep("IntentAdapter", adapter);
        return pipeline;
      },
      metric: "MRR"
    });

    // 今回のケースでは、どのmultiplierでもコサイン類似度でのランクは変わらない（線形スケーリングのため）。
    // ただし、パイプラインが正しく構築され、評価されることを確認する。
    expect(result.allResults.length).toBe(3 * 2); // 3 * 2 = 6 combinations
    expect(result.bestScore).toBeGreaterThan(0);
    expect(result.bestPipeline).toBeInstanceOf(WarpPipeline);
    
    // プログレスコールバックのテスト
    let progressCalls = 0;
    await tuner.tuneGrid({
      searchSpace: { param: [1, 2] },
      pipelineBuilder: () => new WarpPipeline(2),
      onProgress: (cur, tot, best) => {
        progressCalls++;
        expect(tot).toBe(2);
      }
    });
    expect(progressCalls).toBe(2);
  });
});
