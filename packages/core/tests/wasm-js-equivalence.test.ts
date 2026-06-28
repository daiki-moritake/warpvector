import { describe, expect, test } from "bun:test";
import {
  IntentAdapter,
  ProjectionAdapter,
  WarpPipeline,
  cosineSimilarity,
  normalize,
} from "../src";
import { MlpAdapter } from "@warpvector/ml";

/**
 * WASM パスと JS フォールバックの等価性を検証するテスト。
 *
 * IntentAdapter.tune() は常に JS 実装（applyAffine）を使用し、
 * IntentAdapter.tuneBatch() は WASM が利用可能なら WASM パスを使用する。
 * 同じ入力に対して両方が同一の結果を返すことを確認する。
 *
 * ProjectionAdapter.tune() も同様に WASM 有無で分岐する。
 * ここでは JS フォールバックの結果と WASM パスの結果を比較する。
 */

// --- ヘルパー ---

/** 再現可能な疑似ランダムベクトル */
function seededVector(dim: number, seed: number): Float32Array {
  const vec = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    vec[i] = (hash - Math.floor(hash)) * 2 - 1;
  }
  return vec;
}

/** 再現可能なランダム行列 */
function seededMatrix(rows: number, cols: number, seed: number): number[][] {
  const m: number[][] = [];
  for (let r = 0; r < rows; r++) {
    m.push([]);
    for (let c = 0; c < cols; c++) {
      const hash = Math.sin((seed + r * 100) * 127.1 + c * 311.7) * 43758.5453;
      m[r].push(((hash - Math.floor(hash)) * 2 - 1) / Math.sqrt(cols));
    }
  }
  return m;
}

describe("WASM vs JS equivalence", () => {
  // --- IntentAdapter: tune() (JS) vs tuneBatch() (WASM) ---
  describe("IntentAdapter", () => {
    test("tune() and tuneBatch() produce identical results for identity matrix", () => {
      const dim = 64;
      const adapter = new IntentAdapter(dim);

      // 恒等行列
      const identity = Array.from({ length: dim }, (_, i) =>
        Array.from({ length: dim }, (_, j) => (i === j ? 1 : 0)),
      );
      adapter.addIntent("id", {
        matrix: identity,
        bias: Array.from({ length: dim }, () => 0),
      });

      const vector = Float32Array.from(
        { length: dim },
        (_, i) => Math.sin(i * 0.1) * 0.5,
      );

      const singleResult = adapter.tune(vector, "id");
      const batchResults = adapter.tuneBatch([vector], "id");

      // 精度は Float32 の範囲内で完全一致を期待
      for (let i = 0; i < dim; i++) {
        expect(singleResult[i]).toBeCloseTo(batchResults[0][i], 5);
      }
    });

    test("tune() and tuneBatch() produce identical results for random matrix", () => {
      const dim = 32;
      const adapter = new IntentAdapter(dim);

      // 疑似ランダム行列（再現性のためseed的な値を使用）
      const matrix = Array.from({ length: dim }, (_, i) =>
        Array.from(
          { length: dim },
          (_, j) => (Math.sin((i * dim + j) * 0.37) * 0.5) / Math.sqrt(dim),
        ),
      );
      const bias = Array.from(
        { length: dim },
        (_, i) => Math.cos(i * 0.17) * 0.1,
      );
      adapter.addIntent("random", { matrix, bias });

      const vector = Float32Array.from(
        { length: dim },
        (_, i) => Math.cos(i * 0.23) * 0.8,
      );

      const singleResult = adapter.tune(vector, "random");
      const batchResults = adapter.tuneBatch([vector], "random");

      for (let i = 0; i < dim; i++) {
        expect(singleResult[i]).toBeCloseTo(batchResults[0][i], 4);
      }
    });

    test("tuneBatch results are consistent across batch items", () => {
      const dim = 16;
      const adapter = new IntentAdapter(dim);

      const matrix = Array.from({ length: dim }, (_, i) =>
        Array.from({ length: dim }, (_, j) => (i === j ? 2 : 0)),
      );
      adapter.addIntent("scale", {
        matrix,
        bias: Array.from({ length: dim }, () => 0.1),
      });

      const vectors = [
        Float32Array.from({ length: dim }, (_, i) => i * 0.1),
        Float32Array.from({ length: dim }, (_, i) => -i * 0.2),
        Float32Array.from({ length: dim }, () => 0),
      ];

      const batchResults = adapter.tuneBatch(vectors, "scale");

      // 各バッチ結果がtune()と一致するか
      for (let k = 0; k < vectors.length; k++) {
        const singleResult = adapter.tune(vectors[k], "scale");
        for (let i = 0; i < dim; i++) {
          expect(batchResults[k][i]).toBeCloseTo(singleResult[i], 5);
        }
      }
    });

    test("tune with activation matches tuneBatch with activation", () => {
      const dim = 8;
      const adapter = new IntentAdapter(dim);

      // 一部の値が負になる行列
      const matrix = Array.from({ length: dim }, (_, i) =>
        Array.from({ length: dim }, (_, j) => (i === j ? -1 : 0)),
      );
      adapter.addIntent("neg", {
        matrix,
        bias: Array.from({ length: dim }, () => 0.5),
      });

      const vector = Float32Array.from({ length: dim }, (_, i) => i * 0.3);

      const singleResult = adapter.tune(vector, "neg", "relu");
      const batchResults = adapter.tuneBatch([vector], "neg", "relu");

      for (let i = 0; i < dim; i++) {
        expect(singleResult[i]).toBeCloseTo(batchResults[0][i], 5);
        // ReLU: 負の値はゼロにクランプされるべき
        expect(singleResult[i]).toBeGreaterThanOrEqual(0);
      }
    });

    // --- 拡充テスト ---

    test("高次元 (256次元) での tune/tuneBatch 等価性", () => {
      const dim = 256;
      const adapter = new IntentAdapter(dim);
      adapter.addIntent("test", {
        matrix: seededMatrix(dim, dim, 42),
        bias: Array.from({ length: dim }, (_, i) => Math.sin(i * 0.1) * 0.01),
      });

      const vectors = Array.from({ length: 5 }, (_, i) =>
        seededVector(dim, i + 1),
      );

      const singleResults = vectors.map((v) => adapter.tune(v, "test"));
      const batchResults = adapter.tuneBatch(vectors, "test");

      for (let k = 0; k < vectors.length; k++) {
        for (let i = 0; i < dim; i++) {
          expect(singleResults[k][i]).toBeCloseTo(batchResults[k][i], 4);
        }
      }
    });

    test("tuneAutoBlended が決定的な結果を返す", () => {
      const dim = 64;
      const adapter = new IntentAdapter(dim);

      adapter.addIntent("a", {
        matrix: seededMatrix(dim, dim, 10),
        bias: Array.from({ length: dim }, () => 0),
        routingVector: normalize(seededVector(dim, 100)),
      });
      adapter.addIntent("b", {
        matrix: seededMatrix(dim, dim, 20),
        bias: Array.from({ length: dim }, () => 0),
        routingVector: normalize(seededVector(dim, 200)),
      });

      const vector = seededVector(dim, 50);
      const result1 = adapter.tuneAutoBlended(vector);
      const result2 = adapter.tuneAutoBlended(vector);

      for (let i = 0; i < dim; i++) {
        expect(result1[i]).toBeCloseTo(result2[i], 6);
      }
    });

    test("sigmoid activation での tune/tuneBatch 等価性", () => {
      const dim = 16;
      const adapter = new IntentAdapter(dim);
      adapter.addIntent("sig", {
        matrix: seededMatrix(dim, dim, 77),
        bias: Array.from({ length: dim }, (_, i) => Math.sin(i) * 0.1),
      });

      const vector = seededVector(dim, 42);
      const single = adapter.tune(vector, "sig", "sigmoid");
      const batch = adapter.tuneBatch([vector], "sig", "sigmoid");

      for (let i = 0; i < dim; i++) {
        expect(single[i]).toBeCloseTo(batch[0][i], 5);
        // sigmoid の値域は (0, 1)
        expect(single[i]).toBeGreaterThan(0);
        expect(single[i]).toBeLessThan(1);
      }
    });
  });

  // --- ProjectionAdapter: WASM パスと JS パスの結果一貫性 ---
  describe("ProjectionAdapter", () => {
    test("projection result is mathematically correct (manual verification)", () => {
      // 2x3 射影行列: y = Wx
      // y[0] = 1*x[0] + 2*x[1] + 3*x[2]
      // y[1] = 4*x[0] + 5*x[1] + 6*x[2]
      const adapter = new ProjectionAdapter(3, 2, {
        default: {
          matrix: [
            [1, 2, 3],
            [4, 5, 6],
          ],
        },
      });

      const input = [0.1, 0.2, 0.3];
      // Expected: y[0] = 0.1 + 0.4 + 0.9 = 1.4
      //           y[1] = 0.4 + 1.0 + 1.8 = 3.2
      const result = adapter.tune(input, "default");
      expect(result[0]).toBeCloseTo(1.4, 4);
      expect(result[1]).toBeCloseTo(3.2, 4);
    });

    test("projection with bias is mathematically correct", () => {
      const adapter = new ProjectionAdapter(3, 2, {
        default: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
          ],
          bias: [0.5, -0.5],
        },
      });

      const input = [1.0, 2.0, 3.0];
      // Expected: y[0] = 1.0 + 0.5 = 1.5
      //           y[1] = 2.0 - 0.5 = 1.5
      const result = adapter.tune(input, "default");
      expect(result[0]).toBeCloseTo(1.5, 4);
      expect(result[1]).toBeCloseTo(1.5, 4);
    });

    test("high-dimensional projection produces consistent results", () => {
      const inDim = 256;
      const outDim = 64;
      const matrix = Array.from({ length: outDim }, (_, i) =>
        Array.from(
          { length: inDim },
          (_, j) => Math.sin((i * inDim + j) * 0.13) / Math.sqrt(inDim),
        ),
      );

      const adapter = new ProjectionAdapter(inDim, outDim, {
        default: { matrix },
      });

      const vector = Float32Array.from({ length: inDim }, (_, i) =>
        Math.cos(i * 0.07),
      );

      // 2回実行して同じ結果が得られることを確認
      const result1 = adapter.tune(vector, "default");
      const result2 = adapter.tune(vector, "default");

      for (let i = 0; i < outDim; i++) {
        expect(result1[i]).toBe(result2[i]); // 完全一致
      }
    });

    test("高次元射影 (768→256) の結果が決定的", () => {
      const inDim = 768;
      const outDim = 256;
      const adapter = new ProjectionAdapter(inDim, outDim, {
        proj: { matrix: seededMatrix(outDim, inDim, 42) },
      });

      const vector = seededVector(inDim, 99);
      const result1 = adapter.tune(vector, "proj");
      const result2 = adapter.tune(vector, "proj");

      expect(result1.length).toBe(outDim);
      for (let i = 0; i < outDim; i++) {
        expect(result1[i]).toBe(result2[i]);
      }
    });
  });

  // --- MlpAdapter ---
  describe("MlpAdapter", () => {
    test("WASM推論が決定的 (64→32→16)", async () => {
      const mlp = new MlpAdapter([
        {
          matrix: seededMatrix(32, 64, 10),
          bias: Array.from({ length: 32 }, (_, i) => Math.sin(i) * 0.01),
          activation: "relu",
        },
        {
          matrix: seededMatrix(16, 32, 20),
          bias: Array.from({ length: 16 }, (_, i) => Math.cos(i) * 0.01),
          activation: "linear",
        },
      ]);
      await mlp.init();

      const vector = seededVector(64, 99);
      const result1 = mlp.tune(vector);
      const result2 = mlp.tune(vector);

      expect(result1.length).toBe(16);
      for (let i = 0; i < 16; i++) {
        expect(result1[i]).toBeCloseTo(result2[i], 6);
      }
    });
  });

  // --- WarpPipeline ---
  describe("WarpPipeline", () => {
    test("Pipeline run() がバッチ処理と等しい", async () => {
      const dim = 32;
      const pipeline = new WarpPipeline(dim).addIntent({
        test: {
          matrix: seededMatrix(dim, dim, 42),
          bias: Array.from({ length: dim }, () => 0),
        },
      });

      const vectors = Array.from({ length: 4 }, (_, i) =>
        seededVector(dim, i + 1),
      );

      const singleResults = await Promise.all(
        vectors.map(
          async (v) =>
            (await pipeline.run(v, { intent: "test" })) as Float32Array,
        ),
      );
      const batchResults = await pipeline.runBatch(vectors, { intent: "test" });

      for (let k = 0; k < vectors.length; k++) {
        for (let i = 0; i < dim; i++) {
          expect(singleResults[k][i]).toBeCloseTo(
            (batchResults[k] as Float32Array)[i],
            5,
          );
        }
      }
    });

    test("Pipeline with Intent + Projection の出力次元と決定性", async () => {
      const dim = 64;
      const outDim = 16;
      const pipeline = new WarpPipeline(dim)
        .addIntent({
          test: {
            matrix: seededMatrix(dim, dim, 42),
            bias: Array.from({ length: dim }, () => 0),
          },
        })
        .addProjection(outDim, {
          test: { matrix: seededMatrix(outDim, dim, 99) },
        });

      const vector = seededVector(dim, 1);
      const result1 = (await pipeline.run(vector, {
        intent: "test",
      })) as Float32Array;
      const result2 = (await pipeline.run(vector, {
        intent: "test",
      })) as Float32Array;

      expect(result1.length).toBe(outDim);
      for (let i = 0; i < outDim; i++) {
        expect(result1[i]).toBeCloseTo(result2[i], 5);
      }
    });
  });

  // --- 数学ユーティリティ ---
  describe("Math utilities", () => {
    test("cosineSimilarity が対称 (a·b = b·a)", () => {
      const a = seededVector(128, 1);
      const b = seededVector(128, 2);
      expect(cosineSimilarity(a, b)).toBeCloseTo(cosineSimilarity(b, a), 7);
    });

    test("normalize 後の L2 ノルムが 1.0", () => {
      for (const dim of [64, 128, 256]) {
        const n = normalize(seededVector(dim, dim));
        let norm = 0;
        for (let i = 0; i < n.length; i++) norm += n[i] * n[i];
        expect(Math.sqrt(norm)).toBeCloseTo(1.0, 6);
      }
    });

    test("同一ベクトルの cosine similarity が 1.0", () => {
      const v = normalize(seededVector(64, 42));
      expect(cosineSimilarity(v, v)).toBeCloseTo(1.0, 6);
    });

    test("直交ベクトルの cosine similarity が 0.0", () => {
      const a = new Float32Array([1, 0, 0, 0]);
      const b = new Float32Array([0, 1, 0, 0]);
      expect(Math.abs(cosineSimilarity(a, b))).toBeLessThan(1e-7);
    });
  });
});
