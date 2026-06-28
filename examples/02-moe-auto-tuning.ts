/**
 * Cookbook 02: Mixture of Experts (MoE) & Auto-Tuning
 *
 * このスクリプトは、異なるベクトルモデル（例: コサイン類似度ベースとユーグリッド距離ベースなど）
 * のAdapterを動的にルーティングする MoE Adapter の構築と、
 * その最適化を自動で行う PipelineAutoTuner の使い方を示します。
 */

import { WarpPipeline } from "@warpvector/core";
import { MoeAdapter, MlpAdapter } from "@warpvector/ml";
import { PipelineAutoTuner, SearchExample } from "@warpvector/train";

async function run() {
  const dim = 128;

  // 1. エキスパートとなる2つのAdapterを準備する
  // (ここではモックとしてランダムな重みを持つMLPを使います)
  const expert1 = new MlpAdapter([
    {
      matrix: new Float32Array(dim * dim).fill(0.1),
      bias: new Float32Array(dim).fill(0.0),
      activation: "relu",
    },
  ]);

  const expert2 = new MlpAdapter([
    {
      matrix: new Float32Array(dim * dim).fill(-0.1),
      bias: new Float32Array(dim).fill(0.1),
      activation: "linear",
    },
  ]);

  // 2. MoE Adapter を構築
  // 複数のエキスパートを登録し、入力ベクトルとの類似度等でルーティングさせます
  const moeAdapter = new MoeAdapter({
    experts: [
      { id: "expert1", adapter: expert1 },
      { id: "expert2", adapter: expert2 },
    ],
  });

  // 3. AutoML (PipelineAutoTuner) を用いたハイパーパラメータ探索
  // 評価用のデータセット（SearchExample: クエリに対して、どのドキュメントが正解でどれが不正解か）
  const validationData: SearchExample<Float32Array>[] = [
    {
      query: new Float32Array(dim).fill(1.0),
      positive: new Float32Array(dim).fill(1.0), // 正解（同じベクトル）
      negatives: [new Float32Array(dim).fill(-1.0)], // ハズレ
    },
    {
      query: new Float32Array(dim).fill(0.5),
      positive: new Float32Array(dim).fill(0.5), // 正解
      negatives: [new Float32Array(dim).fill(-0.5)], // ハズレ
    },
  ];

  const tuner = new PipelineAutoTuner(validationData);

  // 探索開始（内部で MRR などのメトリクスを用いて評価します）
  const result = await tuner.tuneGrid({
    searchSpace: {
      centroidScale: [0.5, 1.0, 2.0], // 重心のスケールをチューニング対象にする
    },
    pipelineBuilder: (params) => {
      // 与えられたパラメータを用いてパイプラインを動的に構築
      const moeAdapter = new MoeAdapter({
        experts: [
          {
            id: "expert1",
            adapter: expert1,
            centroid: new Float32Array(dim).fill(params.centroidScale),
          },
          {
            id: "expert2",
            adapter: expert2,
            centroid: new Float32Array(dim).fill(-params.centroidScale),
          },
        ],
        routingStrategy: "cosine",
      });
      return new WarpPipeline(dim).addStep("moe", moeAdapter);
    },
  });

  console.log("Auto-Tuning Finished!");
  console.log("Best Configuration Found:", result.bestParams);
  console.log("Best MRR Score:", result.bestScore);

  // 実際の推論 (一番良かったパイプラインを使う)
  const testInput = new Float32Array(dim).fill(1.0);
  const output = await result.bestPipeline.run(testInput);

  console.log("Inference output length:", output.length);
}

run().catch(console.error);
