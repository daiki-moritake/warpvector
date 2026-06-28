/**
 * WarpVector: Vercel Edge / Cloudflare Workers Cookbook
 *
 * このファイルは、Next.js Edge Runtime や Cloudflare Workers といった
 * "エッジ環境"（サーバーレス）で WarpVector を利用するための実践的なテンプレートです。
 *
 * ====================================================================
 * 特徴とベストプラクティス:
 * 1. 【ゼロレイテンシ初期化】
 *    Edge環境のコールドスタートを最小限に抑えるため、モデル（Adapter）の初期化は
 *    リクエストハンドラの「外側」か、最初のリクエスト時に一度だけ行います（Lazy Initialization）。
 * 2. 【グローバルメモリの再利用】
 *    WASMメモリやパイプラインの状態は、Edge環境のコンテキストが生きている間は保持されます。
 * 3. 【ストリーミングと高速推論】
 *    クライアントから送信されたベクトルをリアルタイムで WarpVector パイプラインに通し、
 *    JSONで即座に返却します。
 * ====================================================================
 */

import { WarpPipeline, AdapterRegistry } from "@warpvector/core";
import {
  SafeQuantizationAdapter,
  AnomalyDetectionAdapter,
} from "@warpvector/extras";
import { SoftWhiteningAdapter } from "@warpvector/train";

// プラグイン・アダプターをシステムに登録（AdapterState は string | object のため型アサーションを使用）
AdapterRegistry.registerFinalStage("SafeQuantizationAdapter", (state) =>
  SafeQuantizationAdapter.importState(state as string),
);
AdapterRegistry.register("AnomalyDetectionAdapter", (state) =>
  AnomalyDetectionAdapter.importState(state as string),
);
AdapterRegistry.register("SoftWhiteningAdapter", (state) =>
  SoftWhiteningAdapter.importState(state as string),
);

/**
 * 1. グローバルキャッシュによるパイプラインの保持
 * エッジ関数が破棄されるまでメモリ上にパイプラインを維持します。
 */
let globalPipeline: WarpPipeline | null = null;
const VECTOR_DIMENSION = 1536; // 例: OpenAI text-embedding-3-small

/**
 * 2. パイプラインの遅延初期化 (Lazy Initialization)
 */
async function getPipeline(): Promise<WarpPipeline> {
  if (globalPipeline) {
    return globalPipeline;
  }

  // 初回リクエスト時に一度だけ構築される
  const pipeline = new WarpPipeline(VECTOR_DIMENSION)
    .addStep("anomaly", new AnomalyDetectionAdapter({ maxValue: 3.0 }))
    .addStep(
      "whitening",
      new SoftWhiteningAdapter(VECTOR_DIMENSION, { tau: 0.5 }),
    )
    .setFinalStage(
      "SafeQuantizationAdapter",
      new SafeQuantizationAdapter({
        type: "int8",
        dim: VECTOR_DIMENSION,
        dynamic: true,
        clipThreshold: 127.0,
      }),
    );

  await pipeline.init(); // WASMの初期化など
  globalPipeline = pipeline;

  console.log("WarpVector pipeline successfully initialized on Edge.");
  return globalPipeline;
}

/**
 * 3. Next.js Edge API / Cloudflare Workers 用のハンドラ例
 *
 * @example Request Body
 * {
 *   "vector": [0.1, -0.2, 0.5, ...]
 * }
 */
export default async function handler(req: Request): Promise<Response> {
  // CORS ヘッダーやメソッドチェック（省略可）
  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  try {
    const body = await req.json();
    const inputVector = body.vector;

    if (
      !Array.isArray(inputVector) ||
      inputVector.length !== VECTOR_DIMENSION
    ) {
      return new Response(
        JSON.stringify({
          error: `Invalid vector dimension. Expected ${VECTOR_DIMENSION}`,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }

    // パイプラインの取得（初回のみ初期化され、2回目以降は即座に返る）
    const pipeline = await getPipeline();

    // 4. ベクトルの変換（推論実行）
    const float32Input = new Float32Array(inputVector);
    const optimizedVector = await pipeline.run(float32Input);

    // 5. 結果の返却
    // Float32Array や Int8Array は JSON.stringify の前に通常の配列に変換する
    return new Response(
      JSON.stringify({
        status: "success",
        optimized_vector: Array.from(optimizedVector),
        dimension: optimizedVector.length,
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (error: any) {
    console.error("WarpVector Edge Execution Error:", error);
    return new Response(
      JSON.stringify({
        error: "Internal Server Error",
        details: error.message,
      }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
}
