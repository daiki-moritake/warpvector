import { IntentAdapter, normalize } from "warpvector";

/**
 * サンプル1: ECサイトでの動的コンテキストルーティング
 *
 * ユーザーが「パソコン」と検索したとき、
 * ユーザーのプロファイル（学生 vs ゲーマー）によって検索結果（ベクトル空間）を歪ませます。
 */

// 1. 意図（コンテキスト）ごとの行列とバイアスを定義
// ※実際は Python などで事前学習した重みを JSON でロードします
const userContexts = {
  // 学生（コスパ重視、軽量さ重視）
  student: {
    matrix: [
      [1.5, 0.0, -0.5], // 価格への感度を強調
      [0.0, 1.2, 0.2], // 重量・軽さへの感度を強調
      [-0.2, 0.1, 0.8], // スペックへのこだわりは下げる
    ],
    bias: [0.3, 0.2, -0.1], // 全体的に「安い」「軽い」方向へシフト
  },
  // ゲーマー（ハイスペック重視、価格は気にしない）
  gamer: {
    matrix: [
      [0.2, 0.0, -0.1], // 価格感度を下げる
      [0.0, 0.5, 0.0], // 重量への感度を下げる
      [0.2, -0.1, 1.8], // スペック・グラボ性能への感度を極端に強調
    ],
    bias: [-0.2, -0.1, 0.5], // 全体的に「高性能」方向へシフト
  },
};

// 2. アダプターを初期化
const adapter = new IntentAdapter(userContexts);

// 3. 検索窓に入力された「パソコン」という単語のベースベクトル（例）
const queryVector = [0.1, 0.2, 0.3];

console.log("=== ECサイト: 動的コンテキストルーティング ===");
console.log("ベースの検索ベクトル:", queryVector);

// 4. 学生向けの検索結果に最適化（ワープ）
const studentQuery = adapter.tune(queryVector, "student");
console.log(
  "学生向けにワープされたベクトル:",
  Array.from(normalize(studentQuery)),
);

// 5. ゲーマー向けの検索結果に最適化（ワープ）
const gamerQuery = adapter.tune(queryVector, "gamer");
console.log(
  "ゲーマー向けにワープされたベクトル:",
  Array.from(normalize(gamerQuery)),
);

console.log(
  "\n💡 これにより、同じ「パソコン」という検索でも、全く異なる商品が上位にヒットするようになります！",
);
