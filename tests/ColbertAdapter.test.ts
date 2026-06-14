import { expect, test, describe } from "bun:test";
import { ColbertAdapter } from "../src/ColbertAdapter";
import { initWasm } from "../src/wasm/wasm-loader";

// TypeScript側での参照用実装 (WASMとの結果を比較するため)
function tsColbertMaxSim(query: Float32Array, doc: Float32Array, dim: number): number {
  const queryTokens = query.length / dim;
  const docTokens = doc.length / dim;
  
  let totalScore = 0.0;
  for (let i = 0; i < queryTokens; i++) {
    const qOffset = i * dim;
    let maxSim = -Infinity;
    
    for (let j = 0; j < docTokens; j++) {
      const dOffset = j * dim;
      let sim = 0.0;
      
      for (let k = 0; k < dim; k++) {
        sim += query[qOffset + k] * doc[dOffset + k];
      }
      
      if (sim > maxSim) maxSim = sim;
    }
    totalScore += maxSim;
  }
  
  return totalScore;
}

describe("ColbertAdapter (Late Interaction)", () => {
  test("initWasm", async () => {
    await initWasm();
  });

  const dim = 4;
  // クエリ: 2トークン (2x4)
  const queryTokens = new Float32Array([
    0.1, 0.2, 0.3, 0.4,
    -0.1, 0.5, 0.0, 0.2
  ]);

  // ドキュメント1: 3トークン (3x4)
  const docTokens1 = new Float32Array([
    0.1, 0.2, 0.3, 0.4, // クエリトークン1と完全一致
    0.0, 0.1, 0.1, 0.0,
    -0.1, 0.5, 0.0, 0.2 // クエリトークン2と完全一致
  ]);

  // ドキュメント2: 2トークン (2x4) (無関係なベクトル)
  const docTokens2 = new Float32Array([
    0.0, -0.1, 0.0, 0.0,
    -0.5, 0.0, 0.1, -0.2
  ]);

  test("calculates single MaxSim score correctly matching pure TS implementation", () => {
    const adapter = new ColbertAdapter();

    const wasmScore1 = adapter.score(queryTokens, docTokens1, dim);
    const tsScore1 = tsColbertMaxSim(queryTokens, docTokens1, dim);
    
    // 浮動小数点の誤差を考慮して比較
    expect(wasmScore1).toBeCloseTo(tsScore1, 5);

    const wasmScore2 = adapter.score(queryTokens, docTokens2, dim);
    const tsScore2 = tsColbertMaxSim(queryTokens, docTokens2, dim);
    expect(wasmScore2).toBeCloseTo(tsScore2, 5);
    
    // doc1の方がdoc2より類似度が高くなるはず
    expect(wasmScore1).toBeGreaterThan(wasmScore2);
  });

  test("ranks multiple documents correctly", () => {
    const adapter = new ColbertAdapter();
    const results = adapter.rank(queryTokens, [docTokens1, docTokens2], dim);
    
    expect(results).toHaveLength(2);
    // スコア降順なので、インデックス0(doc1)が先に来るはず
    expect(results[0].index).toBe(0);
    expect(results[1].index).toBe(1);
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });
});
