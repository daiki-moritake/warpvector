/**
 * Cookbook 03: Cross-Encoder Training
 * 
 * このスクリプトは、1st-pass Retrieval (初回のベクトル検索) で得られた
 * 候補ドキュメント群の順位をより高精度に並べ替える (Reranking) ための
 * Cross-Encoder モデルを学習し、ストリーミングデータでOOMを防ぐ方法を示します。
 */

import { CrossEncoderTrainer, CrossEncoderExample } from "@warpvector/train";

// 疑似的なデータストリーム生成関数
// 大規模データの場合、ここをデータベースカーソルやCSVの行単位の読み込みに差し替えます
async function* dataStreamGenerator(): AsyncIterable<CrossEncoderExample> {
  const datasetSize = 10000;
  for (let i = 0; i < datasetSize; i++) {
    // 例として、ランダムなペアを生成
    // 実際には、GPT-4などの強力なLLMが算出した 0.0 ~ 1.0 の関連度を score にセットします
    const isRelated = Math.random() > 0.5;
    yield {
      query: [Math.random(), Math.random(), Math.random()],
      document: isRelated 
        ? [0.9, 0.8, 0.9]  // 関連している場合の仮想特徴
        : [-0.9, -0.8, -0.9], // 関連していない場合
      score: isRelated ? 1.0 : 0.0
    };
  }
}

async function run() {
  const dim = 3;

  // 1. Cross-Encoder Trainer の初期化
  const trainer = new CrossEncoderTrainer(dim, dim);

  console.log("Starting streaming training (this avoids OOM on large datasets)...");

  // 2. メモリに全てロードせず、ストリームから逐次学習する
  // `trainFromGenerator` にデータ生成関数を渡すことで、
  // エポックごとに必要な分だけメモリに展開して最適化を行います。
  const weights = await trainer.trainFromGenerator(dataStreamGenerator, {
    epochs: 5,           // ストリームを5周する
    learningRate: 0.05,
    regularization: 0.01 // 過学習を防ぐためのL2正則化
  });

  console.log("Training complete!");
  const flatMatrix = weights.matrix as Float32Array;
  const biasValue = weights.bias ? weights.bias[0] : 0;
  
  console.log(`Learned weights matrix size: ${flatMatrix.length}`);
  console.log(`Learned bias: ${biasValue}`);

  // 3. 学習済み重みを使って Reranking スコアを計算するデモ
  const newQuery = [0.5, 0.5, 0.5];
  const newDocument = [0.8, 0.9, 0.8]; // おそらく関連度が高いベクトル

  // クエリとドキュメントを連結 (Concatenate)
  const interactionFeature = [...newQuery, ...newDocument];
  
  // 射影行列によるスコア計算 (Affine Transformation: w*x + b)
  let score = biasValue;
  for (let i = 0; i < interactionFeature.length; i++) {
    score += interactionFeature[i] * flatMatrix[i];
  }

  console.log(`Reranking Score for the pair: ${score.toFixed(4)}`);
  // 検索結果に対し、この score が高い順にソート（並び替え）すれば Reranker の完成です。
}

run().catch(console.error);
