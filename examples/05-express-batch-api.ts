/**
 * WarpVector Example 05: Node.js (Express) Streaming API
 * 
 * このサンプルでは、WarpVectorの `runStream` を用いて、
 * クライアントから送信される大量のベクトル（例: 100万件）を
 * メモリを枯渇させることなく（OOMフリーで）安全かつ高速にバッチ変換する
 * 実践的なバックエンドAPIを構築します。
 */

import { WarpPipeline } from "@warpvector/core";
import { AnomalyDetectionAdapter, SafeQuantizationAdapter } from "@warpvector/extras";
import { SoftWhiteningAdapter } from "@warpvector/ml";

const VECTOR_DIMENSION = 1536; // e.g. text-embedding-3-small
let globalPipeline: WarpPipeline | null = null;

async function getPipeline(): Promise<WarpPipeline> {
  if (globalPipeline) return globalPipeline;

  const pipeline = new WarpPipeline(VECTOR_DIMENSION)
    .addStep("anomaly", new AnomalyDetectionAdapter({ maxValue: 3.0 }))
    .addStep("whitening", new SoftWhiteningAdapter(VECTOR_DIMENSION, { tau: 0.5 }))
    .addStep("quantize", new SafeQuantizationAdapter({
      type: "int8",
      dim: VECTOR_DIMENSION,
      dynamic: true,
      clipThreshold: 127.0
    }));

  await pipeline.init(); // Initialize WASM
  globalPipeline = pipeline;
  console.log("Pipeline initialized.");
  return globalPipeline;
}

// === ダミーの Express サーバー構造（コンセプト用） ===
// 実際のプロジェクトでは express と body-parser などをインポートして使用します。

/**
 * 非同期ジェネレータを用いたベクトルのストリーム処理
 * 
 * @param vectorGenerator クライアントから送られてくるベクトルのストリームジェネレータ
 * @param res Expressのレスポンスオブジェクト
 */
export async function handleStreamingVectors(
  vectorGenerator: AsyncGenerator<number[], void, unknown>,
  res: any // Express Response
) {
  try {
    const pipeline = await getPipeline();

    // クライアントに対してストリーミングでレスポンスを返す準備
    res.setHeader("Content-Type", "application/json");
    res.write('{"status":"processing","results":[');

    let isFirst = true;

    // WarpPipeline の runStream を使ってバッチストリーム処理
    // 内部では128件ずつWASM/SIMDで高速処理され、順次 yield されます
    const outStream = pipeline.runStream(vectorGenerator, { batchSize: 128 });

    for await (const outputVector of outStream) {
      if (!isFirst) {
        res.write(",");
      }
      isFirst = false;

      // 結果をチャンクとして送信（JSONストリーミング）
      // Float32Array や Int8Array などを通常の配列に変換して送信
      res.write(JSON.stringify(Array.from(outputVector)));
    }

    res.write("]}");
    res.end();
  } catch (error) {
    console.error("Streaming error:", error);
    res.status(500).json({ error: "Pipeline processing failed." });
  }
}

// テスト用のダミー実行コード
async function runDemo() {
  // 1万件のダミーベクトルを非同期に生成するジェネレータ
  async function* dummyVectorStream() {
    for (let i = 0; i < 10000; i++) {
      const vec = new Array(VECTOR_DIMENSION).fill(Math.random());
      yield vec;
    }
  }

  // ダミーのレスポンスオブジェクト
  const mockRes = {
    setHeader: () => {},
    write: (chunk: string) => process.stdout.write(chunk.length > 50 ? "[chunk]" : chunk),
    end: () => console.log("\n[Stream Ended]"),
    status: () => mockRes,
    json: () => mockRes
  };

  console.log("Starting stream processing demo...");
  await handleStreamingVectors(dummyVectorStream(), mockRes);
}

// コマンドラインから直接実行された場合
if (typeof require !== "undefined" && require.main === module) {
  runDemo().catch(console.error);
}
