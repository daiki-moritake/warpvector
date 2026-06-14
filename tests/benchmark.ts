import { IntentAdapter } from "../src/IntentAdapter";

// ベンチマークのパラメータ
const DIM = 384; // ベクトルの次元数
const BATCH_SIZE = 10000; // 処理するベクトルの数

// ランダムな変換行列とバイアスの生成
const matrix: number[][] = [];
for (let i = 0; i < DIM; i++) {
  const row = [];
  for (let j = 0; j < DIM; j++) row.push(Math.random());
  matrix.push(row);
}
const bias = new Array(DIM).fill(0.1);

// アダプターの初期化
const adapter = new IntentAdapter({
  test: { matrix, bias },
});

// ランダムな入力ベクトルデータの生成
const vectors: number[][] = [];
for (let k = 0; k < BATCH_SIZE; k++) {
  const v = [];
  for (let j = 0; j < DIM; j++) v.push(Math.random());
  vectors.push(v);
}

console.log(
  `Starting benchmark for Batch Size: ${BATCH_SIZE}, Dimension: ${DIM}`,
);

// WASM最適化パスのベンチマーク測定
const startWasm = performance.now();
const resWasm = adapter.tuneBatch(vectors, "test");
const endWasm = performance.now();
console.log(`tuneBatch (WASM): ${(endWasm - startWasm).toFixed(2)} ms`);

// 純粋なJavaScriptループのベンチマーク測定（フォールバック用と同じ処理）
const startJS = performance.now();
const flatMatrix = (adapter as any).matrices.get("test");
const flatBias = (adapter as any).biases.get("test");
const results = new Array<Float32Array>(BATCH_SIZE);
for (let k = 0; k < BATCH_SIZE; k++) {
  const baseVector = vectors[k];
  const result = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) {
    let sum = 0;
    const rowOffset = i * DIM;
    for (let j = 0; j < DIM; j++) {
      sum += flatMatrix[rowOffset + j] * baseVector[j];
    }
    result[i] = sum + flatBias[i];
  }
}
const endJS = performance.now();
console.log(`tuneBatch (Pure JS Loop): ${(endJS - startJS).toFixed(2)} ms`);
