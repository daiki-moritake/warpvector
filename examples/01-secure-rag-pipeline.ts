/**
 * Cookbook 01: Secure RAG Pipeline
 * 
 * このスクリプトは、WarpVector を使って安全かつメモリ効率の良い
 * 検索拡張生成 (RAG) パイプラインを構築する方法を示します。
 * 
 * - AnomalyDetectionAdapter: 悪意のある極端なベクトルやNaNをブロック
 * - SafeQuantizationAdapter: 安全に Int8 量子化を行いメモリを節約
 */

import { WarpPipeline } from "@warpvector/core";
import { AnomalyDetectionAdapter, SafeQuantizationAdapter } from "@warpvector/extras";

async function run() {
  const embeddingDim = 1536; // 例: OpenAI text-embedding-ada-002 の次元数

  // 1. セキュアなパイプラインの定義
  const pipeline = new WarpPipeline(embeddingDim)
    // Step 1: 入力ベクトルのサニタイズ（フェイルセーフモード）
    // 万が一 NaN や 100 を超える異常値が入力された場合でも、ゼロ埋めやクリップで安全に通す
    .addStep("anomaly_detector", new AnomalyDetectionAdapter({
      mode: "safe",
      maxValue: 100.0
    }))
    // Step 2: 最終段で Int8 量子化を行い、DB保存サイズを激減させる
    // SafeQuantizationAdapter は内部で QuantizationAdapter をラップし、
    // オーバーフロー前に再度 -1.0 〜 1.0 にクリッピングしてくれます
    .setFinalStage(new SafeQuantizationAdapter({
      type: "int8",
      dim: embeddingDim,
      clipThreshold: 1.0 // 正規化されたベクトルを想定
    }));

  // 2. パイプラインの初期化（内部でWASMのロードなどが行われます）
  await pipeline.init();

  // 3. ユーザーからの入力ベクトル（例えば悪意あるユーザーがAPIを叩いたとする）
  const maliciousInput = new Float32Array(embeddingDim);
  maliciousInput[0] = NaN;         // 壊れた値
  maliciousInput[1] = Infinity;    // 無限大
  maliciousInput[2] = 50000;       // 極端な外れ値
  maliciousInput[3] = 0.5;         // 正常な値

  console.log("Processing malicious input...");
  
  // 4. パイプライン実行
  const safeOutput = pipeline.run(maliciousInput);

  console.log("Pipeline executed successfully!");
  console.log(`Output type: ${safeOutput.constructor.name}`); // Int8Array
  console.log(`Length: ${safeOutput.length} bytes`);
  
  // 最初の数要素を確認（NaNやInfinityがゼロや最大値にクリップされているか）
  console.log("Safe Output (first 4 elements):", safeOutput.slice(0, 4));

  // この safeOutput を pgvector などのデータベースに安全に保存できます
}

run().catch(console.error);
