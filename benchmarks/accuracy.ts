/**
 * WarpVector 精度ベンチマーク
 *
 * 合成データを使用して、各アダプタの変換精度と性能を検証します。
 * 実行: bun run benchmarks/accuracy.ts
 */
import { IntentAdapter, ProjectionAdapter, WarpPipeline } from "@warpvector/core";
import { QuantizationAdapter } from "@warpvector/extras";
import { MlpAdapter, WhiteningAdapter } from "@warpvector/ml";

// --- ユーティリティ ---

/** ランダムなFloat32ベクトルを生成 */
function randomVector(dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    v[i] = (Math.random() * 2 - 1); // [-1, 1]
  }
  return v;
}

/** ランダムな正規直交行列（近似）を生成 */
function randomMatrix(rows: number, cols: number): number[][] {
  const m: number[][] = [];
  for (let r = 0; r < rows; r++) {
    m.push([]);
    for (let c = 0; c < cols; c++) {
      m[r].push((Math.random() * 2 - 1) / Math.sqrt(cols));
    }
  }
  return m;
}

/** コサイン類似度 */
function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

/** Int8量子化したベクトルのコサイン類似度（復元精度） */
function quantizedCosineSimilarity(
  original: Float32Array,
  quantized: Int8Array,
): number {
  // Int8 → Float32 に戻して比較
  const restored = new Float32Array(quantized.length);
  for (let i = 0; i < quantized.length; i++) {
    restored[i] = quantized[i] / 127;
  }
  return cosineSimilarity(original, restored);
}

// --- ベンチマーク ---

interface BenchmarkResult {
  name: string;
  dimension: string;
  avgTimeUs: number;
  metric?: string;
  metricValue?: number;
}

const results: BenchmarkResult[] = [];

function bench(name: string, fn: () => void, iterations = 1000): number {
  // ウォームアップ
  for (let i = 0; i < 10; i++) fn();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const elapsed = performance.now() - start;
  return (elapsed / iterations) * 1000; // マイクロ秒
}

console.log("=== WarpVector 精度ベンチマーク ===\n");

// --- 1. IntentAdapter (線形変換) ---
{
  const dims = [128, 768, 1536];
  for (const dim of dims) {
    const adapter = new IntentAdapter(dim);
    const matrix = randomMatrix(dim, dim);
    const bias = Array.from({ length: dim }, () => 0);
    adapter.addIntent("bench", { matrix, bias });

    const vector = randomVector(dim);
    const avgTimeUs = bench(`IntentAdapter ${dim}D`, () => {
      adapter.tune(vector, "bench");
    });

    // 精度: 恒等変換の往復テスト
    const identity = new IntentAdapter(dim);
    identity.addIntent("id", {
      matrix: Array.from({ length: dim }, (_, i) =>
        Array.from({ length: dim }, (_, j) => (i === j ? 1 : 0)),
      ),
      bias: Array.from({ length: dim }, () => 0),
    });
    const roundTrip = identity.tune(vector, "id");
    const similarity = cosineSimilarity(vector, roundTrip);

    results.push({
      name: "IntentAdapter",
      dimension: `${dim}D`,
      avgTimeUs,
      metric: "恒等変換精度 (cos sim)",
      metricValue: similarity,
    });
  }
}

// --- 2. ProjectionAdapter (次元削減) ---
{
  const configs = [
    { in: 1536, out: 512 },
    { in: 768, out: 256 },
    { in: 3072, out: 1024 },
  ];

  for (const { in: inDim, out: outDim } of configs) {
    const adapter = new ProjectionAdapter(inDim, outDim, {
      bench: { matrix: randomMatrix(outDim, inDim) },
    });

    const vector = randomVector(inDim);
    const avgTimeUs = bench(`ProjectionAdapter ${inDim}→${outDim}`, () => {
      adapter.tune(vector, "bench");
    });

    results.push({
      name: "ProjectionAdapter",
      dimension: `${inDim}→${outDim}`,
      avgTimeUs,
    });
  }
}

// --- 3. QuantizationAdapter (量子化精度) ---
{
  const dims = [128, 768, 1536];
  for (const dim of dims) {
    const quantizer = new QuantizationAdapter({ type: "int8", dim });
    const vectors = Array.from({ length: 100 }, () => randomVector(dim));

    // 量子化精度の測定
    let totalSimilarity = 0;
    for (const v of vectors) {
      const quantized = quantizer.tune(v) as Int8Array;
      totalSimilarity += quantizedCosineSimilarity(v, quantized);
    }
    const avgSimilarity = totalSimilarity / vectors.length;

    const avgTimeUs = bench(`QuantizationAdapter ${dim}D`, () => {
      quantizer.tune(vectors[0]);
    });

    results.push({
      name: "QuantizationAdapter",
      dimension: `${dim}D (int8)`,
      avgTimeUs,
      metric: "量子化復元精度 (cos sim)",
      metricValue: avgSimilarity,
    });
  }
}

// --- 4. MlpAdapter (WASM推論) ---
{
  const configs = [
    { layers: [{ in: 128, out: 64 }], label: "128→64" },
    { layers: [{ in: 768, out: 256 }], label: "768→256" },
    {
      layers: [
        { in: 1536, out: 512 },
        { in: 512, out: 128 },
      ],
      label: "1536→512→128",
    },
  ];

  for (const config of configs) {
    const mlpLayers = config.layers.map((l, i) => ({
      matrix: randomMatrix(l.out, l.in),
      bias: Array.from({ length: l.out }, () => 0),
      activation: (i < config.layers.length - 1 ? "relu" : "linear") as "relu" | "linear",
    }));

    const mlp = new MlpAdapter(mlpLayers);
    await mlp.init();

    const inputDim = config.layers[0].in;
    const vector = randomVector(inputDim);

    const avgTimeUs = bench(`MlpAdapter ${config.label}`, () => {
      mlp.tune(vector);
    });

    results.push({
      name: "MlpAdapter (WASM)",
      dimension: config.label,
      avgTimeUs,
    });
  }
}

// --- 5. パイプライン統合ベンチマーク ---
{
  const dim = 768;
  const outDim = 256;

  const pipeline = new WarpPipeline(dim)
    .addIntent({
      search: {
        matrix: randomMatrix(dim, dim),
        bias: Array.from({ length: dim }, () => 0),
      },
    })
    .addProjection(outDim, {
      search: { matrix: randomMatrix(outDim, dim) },
    });

  const vector = randomVector(dim);
  const avgTimeUs = bench("Pipeline Intent+Projection", () => {
    pipeline.run(vector, { intent: "search" });
  });

  results.push({
    name: "Pipeline",
    dimension: `${dim}→${outDim} (Intent+Proj)`,
    avgTimeUs,
  });
}

// --- 結果表示 ---
console.log("\n┌─────────────────────────┬──────────────────────┬────────────┬───────────────────────────┬──────────┐");
console.log("│ Adapter                 │ Dimension            │ Avg Time   │ Metric                    │ Value    │");
console.log("├─────────────────────────┼──────────────────────┼────────────┼───────────────────────────┼──────────┤");

for (const r of results) {
  const name = r.name.padEnd(23);
  const dim = r.dimension.padEnd(20);
  const time = `${r.avgTimeUs.toFixed(1)}µs`.padEnd(10);
  const metric = (r.metric || "—").padEnd(25);
  const value = r.metricValue !== undefined ? r.metricValue.toFixed(6).padEnd(8) : "—".padEnd(8);
  console.log(`│ ${name} │ ${dim} │ ${time} │ ${metric} │ ${value} │`);
}

console.log("└─────────────────────────┴──────────────────────┴────────────┴───────────────────────────┴──────────┘");
console.log(`\nTotal benchmarks: ${results.length}`);
