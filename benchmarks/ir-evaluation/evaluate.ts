/**
 * IR精度評価フレームワーク — NDCG@K / MRR@K 計算
 *
 * WarpVector のアダプタ（Intent Warping, Whitening, Quantization等）が
 * 情報検索タスクの精度にどのような影響を与えるかを定量的に評価するフレームワークです。
 *
 * 合成的なIRタスク（クエリとドキュメントの関連性が定義されたデンチマーク）を用いて、
 * NDCG@K（Normalized Discounted Cumulative Gain）と MRR（Mean Reciprocal Rank）を計算します。
 *
 * 実行: bun run benchmarks/ir-evaluation/evaluate.ts
 */
import { IntentAdapter, WarpPipeline } from "@warpvector/core";
import { WhiteningAdapter, IntentMatrixFactory } from "@warpvector/ml";

// --- IR評価メトリクス ---

/**
 * DCG@K (Discounted Cumulative Gain)
 * ランキングの上位K件の関連度スコアを、位置に応じて減衰させて合計する。
 */
function dcgAtK(relevanceScores: number[], k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, relevanceScores.length); i++) {
    dcg += relevanceScores[i] / Math.log2(i + 2);
  }
  return dcg;
}

/**
 * NDCG@K (Normalized DCG)
 * DCG@K を理想的なランキングの DCG@K で割ることで [0, 1] に正規化する。
 * 1.0 が完全なランキング。
 */
function ndcgAtK(relevanceScores: number[], k: number): number {
  const actualDCG = dcgAtK(relevanceScores, k);

  // 理想的なランキング（降順にソート）
  const idealScores = [...relevanceScores].sort((a, b) => b - a);
  const idealDCG = dcgAtK(idealScores, k);

  if (idealDCG === 0) return 0;
  return actualDCG / idealDCG;
}

/**
 * MRR (Mean Reciprocal Rank)
 * 最初の関連ドキュメントの順位の逆数。
 */
function reciprocalRank(relevanceScores: number[]): number {
  for (let i = 0; i < relevanceScores.length; i++) {
    if (relevanceScores[i] > 0) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

// --- 合成IRタスクの生成 ---

/** 再現可能な疑似ランダムベクトル */
function seededVector(dim: number, seed: number): Float32Array {
  const vec = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    vec[i] = (hash - Math.floor(hash)) * 2 - 1;
  }
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

/** 指定した方向にバイアスをかけたベクトル */
function biasedVector(
  dim: number,
  seed: number,
  direction: Float32Array,
  strength: number = 0.7,
): Float32Array {
  const noise = seededVector(dim, seed);
  const vec = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    vec[i] = direction[i] * strength + noise[i] * (1 - strength);
  }
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

function cosSimFloat(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

/**
 * 合成IRタスクを構築する。
 *
 * 2つのドメイン（tech / business）に分かれたドキュメント群を持ち、
 * クエリは特定のドメインのドキュメントを正解とする。
 * Intent Warping の有効性を検証するために、
 * 「ドメインが異なるが表面的に似ているドキュメント」を意図的に含める。
 */
interface IRTask {
  queries: {
    vector: Float32Array;
    targetDomain: string;
    relevanceJudgments: Map<number, number>; // docIndex → relevance (0-2)
  }[];
  documents: {
    vector: Float32Array;
    domain: string;
  }[];
}

function buildSyntheticIRTask(dim: number): IRTask {
  // ドメインごとの方向ベクトル（意図的に相関を高くする）
  const techDirection = seededVector(dim, 42);
  const bizDirection = seededVector(dim, 99);

  // ドキュメント群
  const documents: { vector: Float32Array; domain: string }[] = [];

  // Tech ドキュメント (40個) — 弱いドメインバイアス
  for (let i = 0; i < 40; i++) {
    documents.push({
      vector: biasedVector(dim, 1000 + i, techDirection, 0.15 + (i % 5) * 0.02),
      domain: "tech",
    });
  }

  // Business ドキュメント (40個) — 弱いドメインバイアス
  for (let i = 0; i < 40; i++) {
    documents.push({
      vector: biasedVector(dim, 2000 + i, bizDirection, 0.15 + (i % 5) * 0.02),
      domain: "business",
    });
  }

  // クロスドメインドキュメント (60個) — ドメイン境界上のベクトル
  // Intent Warping なしだと、これらがノイズとして上位に混入する
  for (let i = 0; i < 60; i++) {
    const mixRatio = 0.3 + (i / 60) * 0.4; // 30-70% tech
    const mixedDir = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      mixedDir[d] = techDirection[d] * mixRatio + bizDirection[d] * (1 - mixRatio);
    }
    // 偶数は tech 正解、奇数は business 正解
    documents.push({
      vector: biasedVector(dim, 3000 + i, mixedDir, 0.2),
      domain: i % 2 === 0 ? "tech" : "business",
    });
  }

  // クエリ群 (30クエリ) — ドメインバイアスが弱い
  const queries: IRTask["queries"] = [];
  for (let qi = 0; qi < 15; qi++) {
    // Tech クエリ（ドメインシグナルが弱い）
    const techQuery = biasedVector(dim, 5000 + qi, techDirection, 0.20);
    const techJudgments = new Map<number, number>();
    documents.forEach((doc, idx) => {
      if (doc.domain === "tech") {
        techJudgments.set(idx, 2); // highly relevant
      } else {
        techJudgments.set(idx, 0); // not relevant
      }
    });
    queries.push({
      vector: techQuery,
      targetDomain: "tech",
      relevanceJudgments: techJudgments,
    });

    // Business クエリ（ドメインシグナルが弱い）
    const bizQuery = biasedVector(dim, 6000 + qi, bizDirection, 0.20);
    const bizJudgments = new Map<number, number>();
    documents.forEach((doc, idx) => {
      if (doc.domain === "business") {
        bizJudgments.set(idx, 2);
      } else {
        bizJudgments.set(idx, 0);
      }
    });
    queries.push({
      vector: bizQuery,
      targetDomain: "business",
      relevanceJudgments: bizJudgments,
    });
  }

  return { queries, documents };
}

// --- 評価関数 ---

interface EvalResult {
  method: string;
  dimension: number;
  ndcg10: number;
  ndcg50: number;
  mrr: number;
}

function evaluateRetrieval(
  methodName: string,
  dim: number,
  task: IRTask,
  transformQuery: (q: Float32Array) => Float32Array,
): EvalResult {
  let totalNDCG10 = 0;
  let totalNDCG50 = 0;
  let totalMRR = 0;

  for (const query of task.queries) {
    const transformedQuery = transformQuery(query.vector);

    // 全ドキュメントとの類似度を計算
    const scores = task.documents.map((doc) =>
      cosSimFloat(transformedQuery, doc.vector),
    );

    // スコア順にソートして関連度リストを作成
    const ranked = scores
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score);

    const rankedRelevance = ranked.map(
      (r) => query.relevanceJudgments.get(r.idx) ?? 0,
    );

    totalNDCG10 += ndcgAtK(rankedRelevance, 10);
    totalNDCG50 += ndcgAtK(rankedRelevance, 50);
    totalMRR += reciprocalRank(rankedRelevance);
  }

  const n = task.queries.length;
  return {
    method: methodName,
    dimension: dim,
    ndcg10: totalNDCG10 / n,
    ndcg50: totalNDCG50 / n,
    mrr: totalMRR / n,
  };
}

// --- メイン ---

const DIMS = [64, 256, 768];
const allResults: EvalResult[] = [];

console.log("=== WarpVector IR精度評価 ===");
console.log("合成IRタスク: 2ドメイン (tech/business), 140ドキュメント(+60クロスドメイン), 30クエリ\n");

for (const dim of DIMS) {
  console.log(`--- ${dim}次元 ---`);
  const task = buildSyntheticIRTask(dim);

  // 1. Vanilla（変換なし = ベースライン）
  const vanilla = evaluateRetrieval("Vanilla (no transform)", dim, task, (q) => q);
  allResults.push(vanilla);

  // 2. Intent Warping (手動行列)
  {
    // tech / business それぞれの方向強化行列を手動構築
    const techDirection = seededVector(dim, 42);
    const bizDirection = seededVector(dim, 99);

    const intentAdapter = new IntentAdapter(dim);

    // 簡易的なIntent行列: ドメイン方向を強調するアフィン変換
    const techMatrix = new Float32Array(dim * dim);
    const bizMatrix = new Float32Array(dim * dim);
    for (let i = 0; i < dim; i++) {
      techMatrix[i * dim + i] = 1.0;
      bizMatrix[i * dim + i] = 1.0;
      for (let j = 0; j < dim; j++) {
        techMatrix[i * dim + j] += techDirection[i] * techDirection[j] * 0.5;
        bizMatrix[i * dim + j] += bizDirection[i] * bizDirection[j] * 0.5;
      }
    }

    intentAdapter.addIntent("tech", {
      matrix: techMatrix,
      bias: new Float32Array(dim),
    });
    intentAdapter.addIntent("business", {
      matrix: bizMatrix,
      bias: new Float32Array(dim),
    });

    // クエリのドメインに応じて Intent を選択
    const withIntent = evaluateRetrieval(
      "Intent Warping (manual)",
      dim,
      task,
      (q) => {
        // 簡易的なドメイン判定: tech方向とbiz方向のどちらに近いか
        const techSim = cosSimFloat(q, techDirection);
        const bizSim = cosSimFloat(q, bizDirection);
        return intentAdapter.tune(q, techSim > bizSim ? "tech" : "business");
      },
    );
    allResults.push(withIntent);
  }

  // 3. IntentMatrixFactory (自動生成)
  {
    const factory = new IntentMatrixFactory(dim);

    // tech/business のサンプルベクトルを追加
    const techSamples: Float32Array[] = [];
    const bizSamples: Float32Array[] = [];
    for (let i = 0; i < 5; i++) {
      techSamples.push(task.documents[i].vector);       // 最初の5つは tech
      bizSamples.push(task.documents[50 + i].vector);    // 50番目からは biz
    }

    factory.addCategory("tech", techSamples);
    factory.addCategory("business", bizSamples);

    const autoIntents = await factory.build({
      training: { epochs: 100, learningRate: 0.01, patience: 10 },
    });

    const autoAdapter = new IntentAdapter(dim);
    autoAdapter.addIntent("tech", autoIntents["tech"]);
    autoAdapter.addIntent("business", autoIntents["business"]);

    const withAutoIntent = evaluateRetrieval(
      "IntentMatrixFactory (auto)",
      dim,
      task,
      (q) => {
        return autoAdapter.tuneAutoBlended(q);
      },
    );
    allResults.push(withAutoIntent);
  }

  console.log(`  ✅ ${dim}次元 完了`);
}

// --- 結果出力 ---
console.log("\n=== 結果レポート ===\n");

console.log("| Method | Dim | NDCG@10 | NDCG@50 | MRR |");
console.log("|--------|-----|---------|---------|-----|");

for (const r of allResults) {
  console.log(
    `| ${r.method.padEnd(30)} | ${String(r.dimension).padEnd(4)} | ${(r.ndcg10 * 100).toFixed(1).padStart(6)}% | ${(r.ndcg50 * 100).toFixed(1).padStart(6)}% | ${(r.mrr * 100).toFixed(1).padStart(6)}% |`,
  );
}

// 改善率の計算
console.log("\n### 改善率（Vanilla ベースライン対比）\n");
for (const dim of DIMS) {
  const vanilla = allResults.find(
    (r) => r.method === "Vanilla (no transform)" && r.dimension === dim,
  );
  if (!vanilla) continue;

  console.log(`#### ${dim}次元`);
  const others = allResults.filter(
    (r) => r.method !== "Vanilla (no transform)" && r.dimension === dim,
  );
  for (const r of others) {
    const ndcg10Imp = ((r.ndcg10 - vanilla.ndcg10) / vanilla.ndcg10) * 100;
    const mrrImp = ((r.mrr - vanilla.mrr) / vanilla.mrr) * 100;
    console.log(
      `  ${r.method}: NDCG@10 ${ndcg10Imp >= 0 ? "+" : ""}${ndcg10Imp.toFixed(1)}%, MRR ${mrrImp >= 0 ? "+" : ""}${mrrImp.toFixed(1)}%`,
    );
  }
}

// --- Markdown レポート自動生成 ---
import fs from "fs";
import path from "path";

function generateIRReport(): string {
  const now = new Date().toISOString().split("T")[0];
  const lines: string[] = [];

  lines.push(`# IR精度評価レポート`);
  lines.push(``);
  lines.push(`> 自動生成: ${now} | コーパス: 200ドキュメント (2ドメイン×50 + 60クロス + 40一般) | クエリ: 30件`);
  lines.push(``);
  lines.push(`## 概要`);
  lines.push(``);
  lines.push(`WarpVector の各アダプタが情報検索タスクの精度に与える影響を、`);
  lines.push(`NDCG@K（Normalized Discounted Cumulative Gain）と MRR（Mean Reciprocal Rank）で定量評価しました。`);
  lines.push(``);

  // メインテーブル
  lines.push(`## 結果`);
  lines.push(``);
  lines.push(`| 手法 | 次元 | NDCG@10 | NDCG@50 | MRR |`);
  lines.push(`|------|------|---------|---------|-----|`);

  for (const r of allResults) {
    lines.push(
      `| ${r.method} | ${r.dimension} | ${(r.ndcg10 * 100).toFixed(1)}% | ${(r.ndcg50 * 100).toFixed(1)}% | ${(r.mrr * 100).toFixed(1)}% |`,
    );
  }
  lines.push(``);

  // 改善率テーブル
  lines.push(`## Vanilla ベースライン対比の改善率`);
  lines.push(``);

  for (const dim of DIMS) {
    const vanilla = allResults.find(
      (r) => r.method === "Vanilla (no transform)" && r.dimension === dim,
    );
    if (!vanilla) continue;

    lines.push(`### ${dim}次元`);
    lines.push(``);
    lines.push(`| 手法 | NDCG@10 改善 | MRR 改善 |`);
    lines.push(`|------|-------------|---------|`);

    const others = allResults.filter(
      (r) => r.method !== "Vanilla (no transform)" && r.dimension === dim,
    );
    for (const r of others) {
      const ndcg10Imp = ((r.ndcg10 - vanilla.ndcg10) / vanilla.ndcg10) * 100;
      const mrrImp = ((r.mrr - vanilla.mrr) / vanilla.mrr) * 100;
      const ndcgSign = ndcg10Imp >= 0 ? "+" : "";
      const mrrSign = mrrImp >= 0 ? "+" : "";
      lines.push(
        `| ${r.method} | ${ndcgSign}${ndcg10Imp.toFixed(1)}% | ${mrrSign}${mrrImp.toFixed(1)}% |`,
      );
    }
    lines.push(``);
  }

  // 推奨事項
  lines.push(`## 推奨事項`);
  lines.push(``);

  const intentResult = allResults.find(
    (r) => r.method === "Intent Warping (tech)" && r.dimension === 384,
  );
  const autoResult = allResults.find(
    (r) => r.method.includes("IntentMatrixFactory") && r.dimension === 384,
  );
  const vanillaBase = allResults.find(
    (r) => r.method === "Vanilla (no transform)" && r.dimension === 384,
  );

  if (intentResult && vanillaBase) {
    const imp = ((intentResult.ndcg10 - vanillaBase.ndcg10) / vanillaBase.ndcg10) * 100;
    if (imp > 0) {
      lines.push(`- ✅ **Intent Warping**: NDCG@10 が **+${imp.toFixed(1)}%** 改善。ドメイン特化検索に有効。`);
    }
  }

  if (autoResult && vanillaBase) {
    const imp = ((autoResult.ndcg10 - vanillaBase.ndcg10) / vanillaBase.ndcg10) * 100;
    if (imp > 0) {
      lines.push(`- ✅ **IntentMatrixFactory (auto)**: NDCG@10 が **+${imp.toFixed(1)}%** 改善。手動設定不要。`);
    }
  }

  lines.push(``);
  lines.push(`---`);
  lines.push(``);
  lines.push(`*このレポートは \`bun run benchmarks/ir-evaluation/evaluate.ts\` で自動生成されました。*`);

  return lines.join("\n");
}

const reportPath = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  "REPORT.md",
);
fs.writeFileSync(reportPath, generateIRReport());
console.log(`\n📄 Markdown レポートを生成しました: ${reportPath}`);

