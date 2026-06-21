import { rrf, rsf, RankedResult } from "warpvector";

console.log("=== Hybrid Search with WarpVector ===");

// シミュレーション: Dense Search (WarpVectorでワープさせたベクトル検索の結果)
const denseResults: RankedResult[] = [
  { id: "doc_a", score: 0.95, metadata: { title: "Apple M2 Chip" } }, // 1位
  { id: "doc_b", score: 0.88, metadata: { title: "Banana" } }, // 2位
  { id: "doc_c", score: 0.82, metadata: { title: "Orange" } }, // 3位
  { id: "doc_d", score: 0.75, metadata: { title: "MacBook Pro" } }, // 4位
];

// シミュレーション: Sparse Search (BM25などのキーワード検索の結果)
const sparseResults: RankedResult[] = [
  { id: "doc_d", score: 15.2, metadata: { title: "MacBook Pro" } }, // 1位
  { id: "doc_a", score: 12.1, metadata: { title: "Apple M2 Chip" } }, // 2位
  { id: "doc_e", score: 8.5, metadata: { title: "Pineapple" } }, // 3位
];

console.log("\n--- Method 1: Reciprocal Rank Fusion (RRF) ---");
// RRF は各リストでの「順位」をもとにスコアを再計算します
// スコアの絶対値（0.95 と 15.2 など）が全く異なる場合でも公平に統合できます
const rrfResults = rrf([denseResults, sparseResults]);

console.table(
  rrfResults.map((r, i) => ({
    Rank: i + 1,
    ID: r.id,
    Title: r.metadata?.title,
    "RRF Score": r.score.toFixed(4),
  })),
);

console.log("\n--- Method 2: Relative Score Fusion (RSF) ---");
// RSF は各リストのスコアを [0, 1] の範囲に正規化し、重み付け加算を行います
// 例: Dense 70%, Sparse 30% の割合で検索意図をブレンドしたい場合
const rsfResults = rsf([denseResults, sparseResults], [0.7, 0.3]);

console.table(
  rsfResults.map((r, i) => ({
    Rank: i + 1,
    ID: r.id,
    Title: r.metadata?.title,
    "RSF Score": r.score.toFixed(4),
  })),
);

console.log("\nFusion completes successfully! 🌌");
