import { describe, expect, test } from "bun:test";
import {
  calculateRecall,
  calculateMRR,
  calculateNDCG,
  evaluatePipeline,
  CorpusItem,
  EvalQuery,
} from "../src/evaluator";

describe("Evaluation Metrics", () => {
  test("calculateRecall works correctly", () => {
    // 3つ期待され、2つ見つかった場合 (Recall@3 = 2/3)
    const expected = ["doc1", "doc2", "doc3"];
    const retrieved = ["doc1", "doc4", "doc2"];
    expect(calculateRecall(retrieved, expected, 3)).toBeCloseTo(2 / 3);

    // Kが小さく、1つしか入らない場合 (Recall@1 = 1/3)
    expect(calculateRecall(retrieved, expected, 1)).toBeCloseTo(1 / 3);

    // 存在しない場合
    expect(calculateRecall(["doc5"], expected, 3)).toBe(0);

    // expected が空の場合
    expect(calculateRecall(retrieved, [], 3)).toBe(0);
  });

  test("calculateMRR works correctly", () => {
    const expected = ["doc1", "doc2"];

    // 最初のマッチが1位の場合 (1 / 1 = 1.0)
    expect(calculateMRR(["doc1", "doc3"], expected)).toBe(1.0);

    // 最初のマッチが2位の場合 (1 / 2 = 0.5)
    expect(calculateMRR(["doc3", "doc2", "doc1"], expected)).toBe(0.5);

    // マッチしない場合 (0.0)
    expect(calculateMRR(["doc3", "doc4"], expected)).toBe(0.0);
  });

  test("calculateNDCG works correctly", () => {
    const expected = ["doc1", "doc2"];

    // 全て正解が上位に来た場合 (NDCG = 1.0)
    // DCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    // IDCG = 1/log2(2) + 1/log2(3) = 1.6309
    expect(calculateNDCG(["doc1", "doc2", "doc3"], expected, 3)).toBeCloseTo(
      1.0,
    );

    // 順位が下がった場合 (NDCG < 1.0)
    // Retrieved: ["doc3", "doc1", "doc2"]
    // DCG = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0 + 0.6309 + 0.5 = 1.1309
    // IDCG = 1.6309 (K=3, expected.length=2 => 期待値は2つ全て正解の場合)
    // NDCG = 1.1309 / 1.6309 = 0.6934
    expect(calculateNDCG(["doc3", "doc1", "doc2"], expected, 3)).toBeCloseTo(
      1.13093 / 1.63093,
    );

    // 全くマッチしない場合 (NDCG = 0.0)
    expect(calculateNDCG(["doc3", "doc4"], expected, 3)).toBe(0.0);
  });
});

describe("Pipeline Evaluator Integration", () => {
  const corpus: CorpusItem[] = [
    { id: "doc-1", vector: [1.0, 0.0, 0.0] },
    { id: "doc-2", vector: [0.0, 1.0, 0.0] },
    { id: "doc-3", vector: [0.0, 0.0, 1.0] },
  ];

  const dataset: EvalQuery[] = [
    {
      queryVector: [0.9, 0.1, 0.0],
      expectedDocIds: ["doc-1"],
    },
    {
      queryVector: [0.1, 0.8, 0.1],
      expectedDocIds: ["doc-2"],
    },
  ];

  test("evaluatePipeline compares vanilla and warped results correctly", async () => {
    // パイプラインなし (VanillaとWarpedが同一になるはず)
    const report = await evaluatePipeline({
      corpus,
      dataset,
      kList: [1, 2],
    });

    expect(report.vanilla.mrr).toBeCloseTo(1.0);
    expect(report.warped.mrr).toBeCloseTo(1.0);
    expect(report.vanilla.recall[1]).toBeCloseTo(1.0);
    expect(report.warped.recall[1]).toBeCloseTo(1.0);
    expect(report.vanilla.ndcg[2]).toBeCloseTo(1.0);
    expect(report.warped.ndcg[2]).toBeCloseTo(1.0);
  });

  test("evaluatePipeline works with custom transformation function", async () => {
    // クエリ2 [0.1, 0.8, 0.1] に対して、x方向を大きく引き伸ばす変換を行うと、
    // doc-2 よりも doc-1 の方が類似度が高くなってしまうという悪い変形をモックする
    const badPipeline = (vec: number[] | Float32Array) => {
      const result = new Float32Array(vec);
      result[0] = result[0] * 100; // xを極端に大きくする
      return result;
    };

    const report = await evaluatePipeline({
      corpus,
      dataset,
      kList: [1],
      pipeline: badPipeline,
    });

    // vanilla ではクエリ2は doc-2 (類似度0.8) にマッチして正解するが、
    // warped では x軸方向に引き伸ばされて doc-1 に引っ張られるため、MRRやRecallが低下するはず
    expect(report.vanilla.mrr).toBe(1.0);
    expect(report.warped.mrr).toBeLessThan(1.0); // 性能劣化を検知
  });
});
