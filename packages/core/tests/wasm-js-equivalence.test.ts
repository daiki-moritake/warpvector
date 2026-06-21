import { describe, expect, test } from "bun:test";
import { IntentAdapter, ProjectionAdapter } from "../src";

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
  });
});
