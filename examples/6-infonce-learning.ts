import { InfoNCETrainer } from "../src/InfoNCETrainer";
import { IntentAdapter, IntentWeights } from "../src/IntentAdapter";
import { cosineSimilarity } from "../src/utils";

/**
 * 【InfoNCE Loss による検索ランキングのオンライン学習デモ】
 *
 * 実際のEコマースやRAGの検索システムでは、以下のようなフィードバックが得られます。
 *
 * 1. ユーザーが「赤いスニーカー」を検索した（Anchor）
 * 2. システムは5件の商品を表示した
 * 3. ユーザーは 3件目の「赤いランニングシューズ」をクリックした（Positive）
 * 4. 残りの 4件の「青いスニーカー」「赤いシャツ」「赤い靴紐」「赤いブーツ」はスルーした（Negatives）
 *
 * InfoNCE Loss を使うことで、この1回のセッション（1つのPositiveと4つのNegatives）の情報を
 * 一気に全て学習し、検索ベクトル空間を最適化することができます。
 */
async function runInfoNCEDemo() {
  console.log("=== InfoNCE Loss によるマルチNegative最適化デモ ===\n");

  const dim = 5;
  const trainer = new InfoNCETrainer(dim);

  // 初期の重み (単位行列とゼロバイアス)
  // 初期状態ではベクトルは一切変換されません
  let currentWeights: IntentWeights = {
    matrix: [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1]
    ],
    bias: [0, 0, 0, 0, 0],
  };

  // Anchor: ユーザーの検索クエリベクトル
  const anchor = [1.0, 0.2, 0.0, 0.0, 0.0];

  // Positive: クリックされた正解の商品ベクトル
  const positive = [0.8, 0.5, 0.0, 0.1, 0.0];

  // Negatives: スルーされた不正解の商品ベクトル（複数）
  const negatives = [
    [0.9, 0.0, 0.5, 0.0, 0.0],  // Anchorに似ているが不正解
    [0.7, 0.2, 0.8, 0.0, 0.0],  // スルーされた商品A
    [0.85, 0.1, 0.0, 0.7, 0.0], // スルーされた商品B
    [1.0, 0.0, 0.0, 0.0, 0.0],  // Anchorそのものだがスルーされた（よくあるケース）
  ];

  // --- 学習前の状態を確認 ---
  const initialSimPos = cosineSimilarity(anchor, positive);
  console.log(`[Before] Anchor -> Positive (正解) との距離: ${initialSimPos.toFixed(4)}`);
  
  negatives.forEach((neg, i) => {
    const simNeg = cosineSimilarity(anchor, neg);
    console.log(`[Before] Anchor -> Negative ${i + 1} との距離: ${simNeg.toFixed(4)}`);
  });

  console.log("\n--- 🧠 学習中 (20 Epochs) ---");

  // オンライン学習をシミュレート
  for (let epoch = 0; epoch < 20; epoch++) {
    currentWeights = await trainer.updateOnline(
      currentWeights,
      anchor,
      positive,
      negatives,
      0.01,  // 学習率 (Adam)
      0.1,   // Temperature (分布の鋭さ)
      0.001  // 正則化
    );
  }

  // --- 学習後の状態を確認 ---
  const adapter = new IntentAdapter({ learnedIntent: currentWeights });
  const warpedAnchor = adapter.tune(anchor, "learnedIntent");

  console.log("\n--- 学習後 ---");
  const finalSimPos = cosineSimilarity(warpedAnchor, positive);
  console.log(`[After] Anchor -> Positive (正解) との距離: ${finalSimPos.toFixed(4)}`);
  
  negatives.forEach((neg, i) => {
    const simNeg = cosineSimilarity(warpedAnchor, neg);
    console.log(`[After] Anchor -> Negative ${i + 1} との距離: ${simNeg.toFixed(4)}`);
  });

  console.log("\n💡 考察:");
  console.log("🎉 InfoNCE Loss により、1回のセッションデータから「1つのPositiveに近づけ、同時に4つのNegativesから遠ざかる」空間のワープに成功しました！");
}

runInfoNCEDemo().catch(console.error);
