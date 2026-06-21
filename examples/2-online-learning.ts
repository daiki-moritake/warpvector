import { IntentWeights, IntentTrainer } from "warpvector";

/**
 * サンプル2: ユーザーフィードバックに基づくオンライン学習
 *
 * ユーザーが検索結果をクリック（いいね）した際、その行動履歴を元に
 * フロントエンド（またはエッジ環境）で動的に変換行列を学習・成長させます。
 */

// 1. トレーナーの初期化（次元数: 3）
const trainer = new IntentTrainer(3);

// 現在のユーザーのパーソナライズ重み（初期状態は単位行列とゼロバイアス）
// updateOnline の戻り値である IntentWeights 型を受け取れるように明示的に型定義します
let userWeights: IntentWeights = {
  matrix: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  bias: [0, 0, 0],
};

// ユーザーが検索したキーワードのベクトル
const queryVector = [0.2, 0.5, -0.1];

// ユーザーが実際に「クリック（いいね）」した商品のベクトル（理想の方向）
const clickedItemVector = [0.8, 0.1, 0.3];

console.log("=== オンライン学習 (SGD) ===");
console.log("学習前:");
console.log(" Matrix:", userWeights.matrix);
console.log(" Bias:", userWeights.bias);

// 2. トレーニングループ（実際はユーザーがクリックするたびに1回実行するイメージ）
// updateOnline は非同期関数のため、async 即時関数でラップするか、トップレベル await を使用します
const runTraining = async () => {
  for (let epoch = 1; epoch <= 10; epoch++) {
    // updateOnline で重みを更新します
    // 引数: 現在の重み, クエリベクトル, ターゲットベクトル, 学習率
    userWeights = await trainer.updateOnline(
      userWeights,
      { input: queryVector, target: clickedItemVector },
      { learningRate: 0.05 },
    );

    if (epoch % 2 === 0) {
      console.log(`Epoch ${epoch} - 重みの微調整が完了しました`);
    }
  }

  console.log("\n学習後 (ユーザーの好みに適応完了):");
  console.log(" Matrix:", userWeights.matrix);
  console.log(" Bias:", userWeights.bias);

  console.log(
    "\n💡 次回からこのユーザーの検索ベクトルにこの重みを適用すれば、初めからクリックした商品に近い結果が出やすくなります！",
  );
};

runTraining();
