/**
 * Graceful Degradation テスト
 *
 * WarpVector がエラー状況（不正入力、境界条件、WASM 非対応環境など）で
 * クラッシュせず適切にエラーを報告するか、またはフォールバックするかを検証します。
 */
import { describe, test, expect } from "bun:test";
import {
  IntentAdapter,
  ProjectionAdapter,
  WarpPipeline,
  cosineSimilarity,
  normalize,
  WarpError,
  WarpDimensionMismatchError,
  WarpValidationError,
} from "@warpvector/core";
import { QuantizationAdapter } from "@warpvector/extras";

describe("Graceful Degradation", () => {
  describe("次元不一致のエラーハンドリング", () => {
    test("IntentAdapter に間違った次元のベクトルを渡すとエラー", () => {
      const adapter = new IntentAdapter(64);
      const identity = Array.from({ length: 64 }, (_, i) =>
        Array.from({ length: 64 }, (_, j) => (i === j ? 1 : 0)),
      );
      adapter.addIntent("test", {
        matrix: identity,
        bias: Array.from({ length: 64 }, () => 0),
      });

      // 正しい次元 → 成功
      const correct = adapter.tune(new Float32Array(64), "test");
      expect(correct.length).toBe(64);

      // 間違った次元 → エラー
      expect(() => adapter.tune(new Float32Array(128), "test")).toThrow();
    });

    test("ProjectionAdapter に間違った次元のベクトルを渡すとエラー", () => {
      const adapter = new ProjectionAdapter(3, 2, {
        default: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
          ],
        },
      });

      // 正しい次元 → 成功
      const correct = adapter.tune([1, 2, 3], "default");
      expect(correct.length).toBe(2);

      // 間違った次元 → エラー
      expect(() => adapter.tune([1, 2, 3, 4], "default")).toThrow();
    });

    test("WarpPipeline に間違った次元のベクトルを渡すとエラー", () => {
      const pipeline = new WarpPipeline(3).addIntent({
        test: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
          bias: [0, 0, 0],
        },
      });

      // 正しい次元 → 成功
      const correct = pipeline.run([1, 2, 3], { intent: "test" });
      expect((correct as Float32Array).length).toBe(3);

      // 間違った次元 → エラー
      expect(() => pipeline.run([1, 2, 3, 4], { intent: "test" })).toThrow();
    });
  });

  describe("存在しないインテントのエラーハンドリング", () => {
    test("IntentAdapter に未登録のインテント名を指定するとエラー", () => {
      const adapter = new IntentAdapter(3);
      adapter.addIntent("valid", {
        matrix: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        bias: [0, 0, 0],
      });

      // 存在するインテント → 成功
      expect(() => adapter.tune([1, 2, 3], "valid")).not.toThrow();

      // 存在しないインテント → エラー
      expect(() => adapter.tune([1, 2, 3], "nonexistent")).toThrow();
    });
  });

  describe("空の入力に対する堅牢性", () => {
    test("QuantizationAdapter が空配列を処理してもクラッシュしない", () => {
      const adapter = new QuantizationAdapter({ type: "int8", dim: 0 });
      // dim=0 の場合でもクラッシュせず空の結果を返す
      const result = adapter.encode(new Float32Array(0));
      expect(result.length).toBe(0);
    });

    test("cosineSimilarity が同一ベクトルに 1.0 を返す", () => {
      const v = new Float32Array([0.5, 0.3, 0.8]);
      const sim = cosineSimilarity(v, v);
      expect(sim).toBeCloseTo(1.0, 5);
    });

    test("normalize がゼロベクトルでクラッシュしない", () => {
      const zero = new Float32Array(64);
      // ゼロベクトルの正規化は定義が曖昧だが、クラッシュしないことが重要
      expect(() => normalize(zero)).not.toThrow();
    });
  });

  describe("exportState / importState の堅牢性", () => {
    test("IntentAdapter の状態がラウンドトリップで完全に復元される", () => {
      const adapter = new IntentAdapter(4);
      adapter.addIntent("a", {
        matrix: [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16],
        ],
        bias: [0.1, 0.2, 0.3, 0.4],
      });

      const state = adapter.exportState();
      const json = JSON.stringify(state);
      const restored = IntentAdapter.importState(JSON.parse(json));

      const input = [1.0, 2.0, 3.0, 4.0];
      const original = adapter.tune(input, "a");
      const fromRestored = restored.tune(input, "a");

      for (let i = 0; i < 4; i++) {
        expect(original[i]).toBeCloseTo(fromRestored[i], 6);
      }
    });

    test("WarpPipeline の状態がラウンドトリップで復元される", () => {
      const pipeline = new WarpPipeline(3).addIntent({
        x: {
          matrix: [
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 4],
          ],
          bias: [0.1, 0.2, 0.3],
        },
      });

      const state = pipeline.exportState();
      const json = JSON.stringify(state);
      const restored = WarpPipeline.importState(JSON.parse(json));

      const input = [1, 1, 1];
      const r1 = pipeline.run(input, { intent: "x" }) as Float32Array;
      const r2 = restored.run(input, { intent: "x" }) as Float32Array;

      for (let i = 0; i < 3; i++) {
        expect(r1[i]).toBeCloseTo(r2[i], 6);
      }
    });
  });

  describe("量子化境界条件", () => {
    test("Int8 量子化が極端な値でも安全に動作", () => {
      const adapter = new QuantizationAdapter({ type: "int8", dim: 4 });

      // 大きな値
      const large = new Float32Array([1e6, -1e6, 1e6, -1e6]);
      const result1 = adapter.encode(large);
      expect(result1).toBeInstanceOf(Int8Array);
      expect(result1.length).toBe(4); // dim=4 → 4 bytes Int8

      // 小さな値
      const tiny = new Float32Array([1e-10, -1e-10, 0, 1e-10]);
      const result2 = adapter.encode(tiny);
      expect(result2).toBeInstanceOf(Int8Array);
    });

    test("Binary 量子化が任意の入力で固定サイズの出力を返す", () => {
      const dim = 64;
      const adapter = new QuantizationAdapter({ type: "binary", dim });

      const vec1 = Float32Array.from({ length: dim }, (_, i) => Math.sin(i));
      const result = adapter.encode(vec1);

      // 64次元 → 8バイト (64/8 = 8)
      expect(result).toBeInstanceOf(Uint8Array);
      expect(result.length).toBe(dim / 8);
    });
  });

  describe("構造化エラーの検証", () => {
    test("WarpError の instanceof チェーン", () => {
      const error = new WarpDimensionMismatchError("TestAdapter", 1536, 768);
      expect(error).toBeInstanceOf(WarpError);
      expect(error).toBeInstanceOf(WarpDimensionMismatchError);
      expect(error.code).toBe("DIMENSION_MISMATCH");
      expect(error.expectedDim).toBe(1536);
      expect(error.actualDim).toBe(768);
      expect(error.component).toBe("TestAdapter");
    });

    test("WarpValidationError に正しいフィールド情報が含まれる", () => {
      const error = new WarpValidationError(
        "IntentAdapter",
        "matrix",
        "正方行列が必要です",
      );
      expect(error).toBeInstanceOf(WarpError);
      expect(error.code).toBe("VALIDATION_ERROR");
      expect(error.component).toBe("IntentAdapter");
      expect(error.field).toBe("matrix");
    });
  });
});
