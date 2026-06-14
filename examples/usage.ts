import { IntentAdapter, IntentWeights } from "../src/IntentAdapter";

// 3次元ベクトル空間のダミーインテントを定義
const dummyIntents: Record<string, IntentWeights> = {
  // "friendly" インテント: ベクトルを少し拡大し、ポジティブなバイアスを加算
  friendly: {
    matrix: [
      [1.1, 0.0, 0.0],
      [0.0, 1.1, 0.0],
      [0.0, 0.0, 1.1],
    ],
    bias: [0.5, 0.5, 0.5],
  },
  // "formal" インテント: 特定の方向へのシフトと複雑な変換
  formal: {
    matrix: [
      [0.9, -0.1, 0.0],
      [0.1,  0.9, 0.0],
      [0.0,  0.0, 1.0],
    ],
    bias: [-0.2, 0.0, 0.1],
  },
};

// 1. Adapterの初期化（内部で Float32Array にプリコンパイルされる）
const adapter = new IntentAdapter(dummyIntents);

// 2. 検索クエリやユーザー状態を表すベースベクトル
const baseVector = [1.0, 2.0, 3.0];

console.log("=== Base Vector ===");
console.log(baseVector);
console.log();

// 3. "friendly" 意図の適用
const friendlyVector = adapter.tune(baseVector, "friendly");
console.log("=== Applied 'friendly' intent ===");
// 期待値: W * x + b = [1.1*1 + 0.5, 1.1*2 + 0.5, 1.1*3 + 0.5] = [1.6, 2.7, 3.8]
console.log(friendlyVector);
console.log();

// 4. "formal" 意図の適用
const formalVector = adapter.tune(baseVector, "formal");
console.log("=== Applied 'formal' intent ===");
// 期待値:
// x' = 0.9*1 - 0.1*2 + 0.0*3 - 0.2 = 0.9 - 0.2 - 0.2 = 0.5
// y' = 0.1*1 + 0.9*2 + 0.0*3 + 0.0 = 0.1 + 1.8 + 0.0 = 1.9
// z' = 0.0*1 + 0.0*2 + 1.0*3 + 0.1 = 3.0 + 0.1 = 3.1
console.log(formalVector);
console.log();

// 5. Float32Array を入力として用いる（より高速な計算）
const float32Base = new Float32Array([1.0, 2.0, 3.0]);
console.time("Tune friendly");
const fastResult = adapter.tune(float32Base, "friendly");
console.timeEnd("Tune friendly");

console.log("\nTransformation completed successfully!");
