/**
 * 量子化品質ベンチマーク — Recall@K 比較
 *
 * Float32（原点）に対する各量子化方式の Recall@K を測定します。
 * 「量子化しても、元のFloat32検索と同じ結果が返ってくるか？」を定量的に評価します。
 *
 * 実行: bun run benchmarks/quantization-quality/recall-at-k.ts
 */
import { QuantizationAdapter } from "@warpvector/extras";

// --- 設定 ---
const DIMENSIONS = [128, 768, 1536];
const NUM_QUERIES = 50;
const NUM_CORPUS = 500;
const K_VALUES = [1, 5, 10, 50, 100];

// --- ユーティリティ ---

/** 再現可能な疑似ランダムベクトル生成 */
function seededVector(dim: number, seed: number): Float32Array {
  const vec = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    vec[i] = (hash - Math.floor(hash)) * 2 - 1;
  }
  // L2 正規化
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

/** Float32ベクトル同士のコサイン類似度 */
function cosSimFloat(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

/** Int8量子化ベクトル同士の近似コサイン類似度（Int32で内積） */
function cosSimInt8(a: Int8Array, b: Int8Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

/** Binary量子化ベクトル同士のハミング距離ベースの類似度 */
function hammingSimilarity(a: Uint8Array, b: Uint8Array): number {
  let matchingBits = 0;
  const totalBits = a.length * 8;
  for (let i = 0; i < a.length; i++) {
    const xor = a[i] ^ b[i];
    // popcount
    let count = xor;
    count = count - ((count >> 1) & 0x55);
    count = (count & 0x33) + ((count >> 2) & 0x33);
    count = (count + (count >> 4)) & 0x0f;
    matchingBits += 8 - count;
  }
  return matchingBits / totalBits;
}

/**
 * Top-K の検索結果を返す（降順ソート）
 */
function topK(
  query: number,
  scores: number[],
  k: number,
): number[] {
  return scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((x) => x.idx);
}

/**
 * Recall@K を計算
 * ground_truth: Float32での正解上位K件のインデックス集合
 * retrieved: 量子化後の上位K件のインデックス集合
 */
function recallAtK(groundTruth: number[], retrieved: number[]): number {
  const gtSet = new Set(groundTruth);
  let hits = 0;
  for (const idx of retrieved) {
    if (gtSet.has(idx)) hits++;
  }
  return hits / groundTruth.length;
}

// --- メイン ---

interface RecallResult {
  method: string;
  dimension: number;
  k: number;
  recall: number;
  avgCosineSim: number;
  sizeReduction: string;
}

const allResults: RecallResult[] = [];

console.log("=== WarpVector 量子化品質ベンチマーク ===");
console.log(`コーパスサイズ: ${NUM_CORPUS}, クエリ数: ${NUM_QUERIES}\n`);

for (const dim of DIMENSIONS) {
  console.log(`--- ${dim}次元 ---`);

  // コーパスとクエリの生成（再現可能）
  const corpus: Float32Array[] = [];
  for (let i = 0; i < NUM_CORPUS; i++) {
    corpus.push(seededVector(dim, i * 7 + 1));
  }
  const queries: Float32Array[] = [];
  for (let i = 0; i < NUM_QUERIES; i++) {
    queries.push(seededVector(dim, 10000 + i * 13));
  }

  // === Int8 量子化 ===
  {
    const quantizer = new QuantizationAdapter({ type: "int8", dim });

    // コーパスを量子化
    const quantizedCorpus: Int8Array[] = corpus.map(
      (v) => quantizer.encode(v) as Int8Array,
    );

    // コサイン類似度の復元精度を測定
    let totalCosSim = 0;
    let numPairs = 0;

    for (const k of K_VALUES) {
      let totalRecall = 0;

      for (let qi = 0; qi < NUM_QUERIES; qi++) {
        const queryVec = queries[qi];
        const quantizedQuery = quantizer.encode(queryVec) as Int8Array;

        // Float32 での正解スコア
        const f32Scores = corpus.map((c) => cosSimFloat(queryVec, c));
        const groundTruth = topK(qi, f32Scores, k);

        // Int8 での近似スコア
        const int8Scores = quantizedCorpus.map((c) =>
          cosSimInt8(quantizedQuery, c),
        );
        const retrieved = topK(qi, int8Scores, k);

        totalRecall += recallAtK(groundTruth, retrieved);

        // コサイン復元精度（k=10のときだけ計測）
        if (k === 10) {
          for (let ci = 0; ci < Math.min(50, NUM_CORPUS); ci++) {
            const orig = cosSimFloat(queryVec, corpus[ci]);
            const approx = cosSimInt8(quantizedQuery, quantizedCorpus[ci]);
            totalCosSim += 1 - Math.abs(orig - approx);
            numPairs++;
          }
        }
      }

      const avgRecall = totalRecall / NUM_QUERIES;
      allResults.push({
        method: "Int8",
        dimension: dim,
        k,
        recall: avgRecall,
        avgCosineSim: numPairs > 0 ? totalCosSim / numPairs : 0,
        sizeReduction: "75%",
      });
    }
  }

  // === Binary 量子化 ===
  {
    const quantizer = new QuantizationAdapter({ type: "binary", dim });

    const quantizedCorpus: Uint8Array[] = corpus.map(
      (v) => quantizer.encode(v) as Uint8Array,
    );

    let totalCosSim = 0;
    let numPairs = 0;

    for (const k of K_VALUES) {
      let totalRecall = 0;

      for (let qi = 0; qi < NUM_QUERIES; qi++) {
        const queryVec = queries[qi];
        const quantizedQuery = quantizer.encode(queryVec) as Uint8Array;

        // Float32 での正解スコア
        const f32Scores = corpus.map((c) => cosSimFloat(queryVec, c));
        const groundTruth = topK(qi, f32Scores, k);

        // Binary でのハミング類似度
        const binScores = quantizedCorpus.map((c) =>
          hammingSimilarity(quantizedQuery, c),
        );
        const retrieved = topK(qi, binScores, k);

        totalRecall += recallAtK(groundTruth, retrieved);

        if (k === 10) {
          for (let ci = 0; ci < Math.min(50, NUM_CORPUS); ci++) {
            const orig = cosSimFloat(queryVec, corpus[ci]);
            const approx = hammingSimilarity(quantizedQuery, quantizedCorpus[ci]) * 2 - 1;
            totalCosSim += 1 - Math.abs(orig - approx);
            numPairs++;
          }
        }
      }

      const avgRecall = totalRecall / NUM_QUERIES;
      allResults.push({
        method: "Binary",
        dimension: dim,
        k,
        recall: avgRecall,
        avgCosineSim: numPairs > 0 ? totalCosSim / numPairs : 0,
        sizeReduction: "96.9%",
      });
    }
  }

  console.log(`  ✅ ${dim}次元: Int8 & Binary 完了`);
}

// --- 結果出力 ---
console.log("\n=== 結果レポート ===\n");

// テーブル形式で出力
console.log("### Recall@K 比較表\n");
console.log("| Method | Dim | K | Recall@K | Size Reduction |");
console.log("|--------|-----|---|----------|----------------|");

for (const r of allResults) {
  console.log(
    `| ${r.method.padEnd(6)} | ${String(r.dimension).padEnd(4)} | ${String(r.k).padEnd(3)} | ${(r.recall * 100).toFixed(1).padStart(6)}% | ${r.sizeReduction.padEnd(14)} |`,
  );
}

// サマリー
console.log("\n### サマリー\n");
for (const dim of DIMENSIONS) {
  console.log(`#### ${dim}次元`);
  for (const method of ["Int8", "Binary"]) {
    const r10 = allResults.find(
      (r) => r.method === method && r.dimension === dim && r.k === 10,
    );
    const r100 = allResults.find(
      (r) => r.method === method && r.dimension === dim && r.k === 100,
    );
    if (r10 && r100) {
      console.log(
        `  ${method}: Recall@10 = ${(r10.recall * 100).toFixed(1)}%, Recall@100 = ${(r100.recall * 100).toFixed(1)}%, CosSim精度 = ${(r10.avgCosineSim * 100).toFixed(1)}%`,
      );
    }
  }
}
