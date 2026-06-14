import { TripletTrainer, IntentAdapter, IntentWeights } from "../src";

// ユークリッド距離（L2）を計算するヘルパー
function euclideanDistance(v1: Float32Array | number[], v2: Float32Array | number[]): number {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += Math.pow(v1[i] - v2[i], 2);
  }
  return Math.sqrt(sum);
}

// 単位行列を作成するヘルパー
function identityWeights(dim: number): IntentWeights {
  const matrix = Array(dim).fill(0).map((_, i) => 
    Array(dim).fill(0).map((_, j) => (i === j ? 1.0 : 0.0))
  );
  return {
    matrix,
    bias: Array(dim).fill(0),
  };
}

async function main() {
  console.log("=== R&D: トリプレットロス (Contrastive Learning) によるエッジ学習 ===");

  const DIM = 3; // わかりやすいように3次元
  const trainer = new TripletTrainer(DIM);

  // 検索クエリ (Anchor) "ノートパソコン"
  const anchor = [0.1, 0.9, 0.1];

  // ユーザーがクリックした商品 (Positive) "ゲーミングPC"
  const positive = [0.8, 0.8, 0.2];

  // ユーザーがスルーした商品 (Negative) "ビジネス用PC"
  const negative = [0.2, 0.9, 0.8];

  // 初期状態の重み（恒等行列：何も変換しない状態）
  let weights = identityWeights(DIM);

  // ----------------------------------------------------
  // 学習前の距離を確認
  // ----------------------------------------------------
  console.log("\n--- 学習前 ---");
  const initialAdapter = new IntentAdapter({ default: weights });
  const warpedAnchorBefore = initialAdapter.tune(anchor, "default");
  
  const distPosBefore = euclideanDistance(warpedAnchorBefore, positive);
  const distNegBefore = euclideanDistance(warpedAnchorBefore, negative);
  
  console.log(`[Before] Anchor -> Positive (正解) との距離: ${distPosBefore.toFixed(4)}`);
  console.log(`[Before] Anchor -> Negative (不正解) との距離: ${distNegBefore.toFixed(4)}`);

  // ----------------------------------------------------
  // トリプレットロスを用いたオンライン学習（フィードバックループ）
  // ----------------------------------------------------
  console.log("\n--- 🧠 学習中 (100 Epochs) ---");
  const EPOCHS = 100;
  const LEARNING_RATE = 0.05;
  const MARGIN = 0.2; // PositiveとNegativeの間に最低限欲しいマージン
  
  for (let i = 0; i < EPOCHS; i++) {
    weights = await trainer.updateOnline(
      weights, 
      anchor, 
      positive, 
      negative, 
      LEARNING_RATE,
      MARGIN
    );
  }

  // ----------------------------------------------------
  // 学習後の距離を確認
  // ----------------------------------------------------
  console.log("\n--- 学習後 ---");
  const learnedAdapter = new IntentAdapter({ default: weights });
  const warpedAnchorAfter = learnedAdapter.tune(anchor, "default");
  
  const distPosAfter = euclideanDistance(warpedAnchorAfter, positive);
  const distNegAfter = euclideanDistance(warpedAnchorAfter, negative);
  
  console.log(`[After] Anchor -> Positive (正解) との距離: ${distPosAfter.toFixed(4)}`);
  console.log(`[After] Anchor -> Negative (不正解) との距離: ${distNegAfter.toFixed(4)}`);

  console.log("\n💡 考察:");
  if (distPosAfter < distPosBefore && distNegAfter > distNegBefore) {
    console.log("🎉 成功！アンカーが正解（Positive）に近づき、不正解（Negative）から遠ざかるように空間が歪められました。");
  } else {
    console.log("⚠️ 学習率やマージンなどのハイパーパラメータ調整が必要です。");
  }
}

main().catch(console.error);
