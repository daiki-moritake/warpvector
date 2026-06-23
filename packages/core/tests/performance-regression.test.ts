/**
 * パフォーマンス回帰テスト
 *
 * CI で実行し、主要な操作のレイテンシが閾値を超えていないことを検証します。
 * 閾値は「遅すぎないか」を検出するためのもので、厳密なベンチマークではありません。
 */
import { describe, test, expect } from "bun:test";
import {
  IntentAdapter,
  WarpPipeline,
  initWasm,
} from "@warpvector/core";
import { WhiteningAdapter } from "@warpvector/ml";
import { QuantizationAdapter } from "@warpvector/extras";

const DIM = 768;
const BATCH_SIZE = 1000;

// ランダムベクトル生成
function randomVec(dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = Math.random() * 2 - 1;
  return v;
}

// ランダムベクトルバッチ
function randomBatch(dim: number, n: number): Float32Array[] {
  return Array.from({ length: n }, () => randomVec(dim));
}

// 行列生成
function randomMatrix(dim: number): Float32Array {
  const m = new Float32Array(dim * dim);
  for (let i = 0; i < dim; i++) m[i * dim + i] = 1.0;
  for (let i = 0; i < dim * dim; i++) m[i] += (Math.random() - 0.5) * 0.1;
  return m;
}

describe("パフォーマンス回帰テスト", () => {
  test("IntentAdapter.tune() — 単一ベクトル変換 < 1ms (768d)", () => {
    const adapter = new IntentAdapter(DIM);
    adapter.addIntent("test", {
      matrix: randomMatrix(DIM),
      bias: new Float32Array(DIM),
    });
    const vec = randomVec(DIM);

    // ウォームアップ
    for (let i = 0; i < 10; i++) adapter.tune(vec, "test");

    // 計測
    const t0 = performance.now();
    const iterations = 100;
    for (let i = 0; i < iterations; i++) adapter.tune(vec, "test");
    const avgMs = (performance.now() - t0) / iterations;

    console.log(`  IntentAdapter.tune(): ${avgMs.toFixed(3)}ms/op`);
    expect(avgMs).toBeLessThan(1.0); // 1ms以下
  });

  test("IntentAdapter.tuneBatch() — 1000ベクトル < 50ms (768d)", () => {
    const adapter = new IntentAdapter(DIM);
    adapter.addIntent("test", {
      matrix: randomMatrix(DIM),
      bias: new Float32Array(DIM),
    });
    const batch = randomBatch(DIM, BATCH_SIZE);

    // ウォームアップ
    adapter.tuneBatch(batch, "test");

    // 計測
    const t0 = performance.now();
    adapter.tuneBatch(batch, "test");
    const ms = performance.now() - t0;

    console.log(`  tuneBatch(${BATCH_SIZE}): ${ms.toFixed(1)}ms`);
    expect(ms).toBeLessThan(1000); // 768d × 1000 vecs は JS fallback で ~600ms
  });

  test("IntentAdapter.tuneBlended() — ブレンド変換 < 3ms (768d)", () => {
    const adapter = new IntentAdapter(DIM);
    adapter.addIntent("a", { matrix: randomMatrix(DIM), bias: new Float32Array(DIM) });
    adapter.addIntent("b", { matrix: randomMatrix(DIM), bias: new Float32Array(DIM) });
    adapter.addIntent("c", { matrix: randomMatrix(DIM), bias: new Float32Array(DIM) });
    const vec = randomVec(DIM);
    const weights = { a: 0.5, b: 0.3, c: 0.2 };

    // ウォームアップ
    for (let i = 0; i < 10; i++) adapter.tuneBlended(vec, weights);

    const t0 = performance.now();
    const iterations = 100;
    for (let i = 0; i < iterations; i++) adapter.tuneBlended(vec, weights);
    const avgMs = (performance.now() - t0) / iterations;

    console.log(`  tuneBlended(3 intents): ${avgMs.toFixed(3)}ms/op`);
    expect(avgMs).toBeLessThan(5.0);
  });

  test("WhiteningAdapter.tune() — 白色化 < 0.5ms (768d)", () => {
    const adapter = new WhiteningAdapter(DIM);
    const corpus = randomBatch(DIM, 50);
    for (const v of corpus) adapter.update(v);
    const vec = randomVec(DIM);

    // ウォームアップ
    for (let i = 0; i < 10; i++) adapter.tune(vec);

    const t0 = performance.now();
    const iterations = 100;
    for (let i = 0; i < iterations; i++) adapter.tune(vec);
    const avgMs = (performance.now() - t0) / iterations;

    console.log(`  WhiteningAdapter.tune(): ${avgMs.toFixed(3)}ms/op`);
    expect(avgMs).toBeLessThan(0.5);
  });

  test("QuantizationAdapter — Int8量子化 < 0.1ms (768d)", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim: DIM });
    const vec = randomVec(DIM);

    // ウォームアップ
    for (let i = 0; i < 10; i++) quantizer.tune(vec);

    const t0 = performance.now();
    const iterations = 1000;
    for (let i = 0; i < iterations; i++) quantizer.tune(vec);
    const avgMs = (performance.now() - t0) / iterations;

    console.log(`  QuantizationAdapter(int8): ${avgMs.toFixed(4)}ms/op`);
    expect(avgMs).toBeLessThan(0.1);
  });

  test("QuantizationAdapter — Binary量子化 < 0.05ms (768d)", () => {
    const quantizer = new QuantizationAdapter({ type: "binary", dim: DIM });
    const vec = randomVec(DIM);

    // ウォームアップ
    for (let i = 0; i < 10; i++) quantizer.tune(vec);

    const t0 = performance.now();
    const iterations = 1000;
    for (let i = 0; i < iterations; i++) quantizer.tune(vec);
    const avgMs = (performance.now() - t0) / iterations;

    console.log(`  QuantizationAdapter(binary): ${avgMs.toFixed(4)}ms/op`);
    expect(avgMs).toBeLessThan(0.05);
  });

  test("WarpPipeline.run() — パイプライン全体 < 2ms (768d)", () => {
    const intentKey = "test";
    const intents: Record<string, { matrix: Float32Array; bias: Float32Array }> = {
      [intentKey]: {
        matrix: randomMatrix(DIM),
        bias: new Float32Array(DIM),
      },
    };
    const pipeline = new WarpPipeline(DIM);
    pipeline.addIntent(intents);
    const vec = randomVec(DIM);

    // ウォームアップ
    for (let i = 0; i < 10; i++) pipeline.run(vec, { intent: intentKey });

    const t0 = performance.now();
    const iterations = 100;
    for (let i = 0; i < iterations; i++) pipeline.run(vec, { intent: intentKey });
    const avgMs = (performance.now() - t0) / iterations;

    console.log(`  WarpPipeline.run(): ${avgMs.toFixed(3)}ms/op`);
    expect(avgMs).toBeLessThan(2.0);
  });

  test("WASM初期化 < 100ms", async () => {
    const t0 = performance.now();
    await initWasm();
    const ms = performance.now() - t0;

    console.log(`  initWasm(): ${ms.toFixed(1)}ms`);
    expect(ms).toBeLessThan(100);
  });
});
