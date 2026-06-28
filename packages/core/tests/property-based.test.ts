import { describe, it, expect } from "bun:test";
import {
  IntentAdapter,
  LoraIntentAdapter,
  ProjectionAdapter,
  WarpPipeline,
} from "@warpvector/core";
import {
  MlpAdapter,
  WhiteningAdapter,
} from "@warpvector/ml";
import {
  QuantizationAdapter,
  VsaAdapter,
} from "@warpvector/extras";
import { ColbertAdapter } from "@warpvector/rerank";

/**
 * プロパティベーステスト: Serialization Roundtrip
 *
 * 全てのアダプタとパイプラインに対して、
 * exportState → importState のラウンドトリップで元の出力が完全に再現されることを検証する。
 * これはインフラとしての信頼性を担保するための最も重要なテストカテゴリ。
 */

/** テスト用ベクトルの生成 */
function testVector(dim: number, seed: number = 42): Float32Array {
  const vec = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    vec[i] = (hash - Math.floor(hash)) * 2 - 1; // [-1, 1]
  }
  return vec;
}

/** テスト用の行列を生成（単位行列+ノイズ） */
function testMatrix(dim: number, seed: number = 1): Float32Array {
  const mat = new Float32Array(dim * dim);
  for (let i = 0; i < dim; i++) {
    mat[i * dim + i] = 1.0;
  }
  for (let i = 0; i < dim * dim; i++) {
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    mat[i] += (hash - Math.floor(hash) - 0.5) * 0.1;
  }
  return mat;
}

/** 2つのベクトルの最大絶対差を計算 */
function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let max = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) max = diff;
  }
  return max;
}

describe("プロパティベーステスト: Serialization Roundtrip", () => {
  const dim = 16;
  const epsilon = 1e-5;

  it("IntentAdapter: exportState → importState で出力が再現される", () => {
    const adapter = new IntentAdapter(dim);
    adapter.addIntent("tech", {
      matrix: testMatrix(dim, 1),
      bias: testVector(dim, 2),
    });
    adapter.addIntent("biz", {
      matrix: testMatrix(dim, 3),
      bias: testVector(dim, 4),
    });

    const stateJson = adapter.exportState();
    const restored = IntentAdapter.importState(stateJson);

    const input = testVector(dim, 99);
    const original = adapter.tune(input, "tech");
    const roundtripped = restored.tune(input, "tech");

    expect(maxAbsDiff(original, roundtripped)).toBeLessThan(epsilon);
  });

  it("LoraIntentAdapter: exportState → importState で出力が再現される", () => {
    const rank = 4;
    // matrixA: [dim][rank] (2D配列)
    const matrixA: number[][] = [];
    for (let i = 0; i < dim; i++) {
      const row: number[] = [];
      for (let j = 0; j < rank; j++) {
        row.push(Math.sin((i * rank + j) * 0.1) * 0.1);
      }
      matrixA.push(row);
    }
    // matrixB: [rank][dim] (2D配列)
    const matrixB: number[][] = [];
    for (let i = 0; i < rank; i++) {
      const row: number[] = [];
      for (let j = 0; j < dim; j++) {
        row.push(Math.cos((i * dim + j) * 0.1) * 0.1);
      }
      matrixB.push(row);
    }
    const bias = Array.from({ length: dim }, (_, i) => Math.sin(i) * 0.01);

    const adapter = new LoraIntentAdapter(dim, rank);
    adapter.addIntent("test", { matrixA, matrixB, bias });

    const stateJson = adapter.exportState();
    const restored = LoraIntentAdapter.importState(stateJson);

    const input = testVector(dim, 42);
    const original = adapter.tune(input, "test");
    const roundtripped = restored.tune(input, "test");

    expect(maxAbsDiff(original, roundtripped)).toBeLessThan(epsilon);
  });

  it("ProjectionAdapter: exportState → importState で出力が再現される", () => {
    const outDim = 8;
    const matrix = new Float32Array(outDim * dim);
    for (let i = 0; i < outDim * dim; i++) {
      matrix[i] = Math.sin(i * 0.3) * 0.2;
    }
    const bias = new Float32Array(outDim);
    for (let i = 0; i < outDim; i++) bias[i] = i * 0.01;

    const adapter = new ProjectionAdapter(dim, outDim, {
      default: { matrix, bias },
    });

    const stateJson = adapter.exportState();
    const restored = ProjectionAdapter.importState(stateJson);

    const input = testVector(dim, 77);
    const original = adapter.tune(input, "default");
    const roundtripped = restored.tune(input, "default");

    expect(maxAbsDiff(original, roundtripped)).toBeLessThan(epsilon);
  });

  it("QuantizationAdapter (Int8): exportState → importState で出力が再現される", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim });
    const stateJson = quantizer.exportState();
    const restored = QuantizationAdapter.importState(stateJson as string);

    const input = testVector(dim, 55);
    const original = quantizer.encode(input);
    const roundtripped = restored.encode(input);

    expect(original.length).toBe(roundtripped.length);
    expect(maxAbsDiff(original, roundtripped)).toBe(0);
  });

  it("QuantizationAdapter (Binary): exportState → importState で出力が再現される", () => {
    const quantizer = new QuantizationAdapter({ type: "binary", dim });
    const stateJson = quantizer.exportState();
    const restored = QuantizationAdapter.importState(stateJson as string);

    const input = testVector(dim, 88);
    const original = quantizer.encode(input);
    const roundtripped = restored.encode(input);

    expect(original.length).toBe(roundtripped.length);
    expect(maxAbsDiff(original, roundtripped)).toBe(0);
  });

  it("WarpPipeline: exportState → importState で出力が再現される", async () => {
    const pipeline = new WarpPipeline(dim).addIntent({
      tech: { matrix: testMatrix(dim, 1), bias: testVector(dim, 2) },
    });

    const state = pipeline.exportState();
    const restored = WarpPipeline.importState(state);

    const input = testVector(dim, 33);
    const original = await pipeline.run(input, { intent: "tech" });
    const roundtripped = await restored.run(input, { intent: "tech" });

    expect(
      maxAbsDiff(original as Float32Array, roundtripped as Float32Array),
    ).toBeLessThan(epsilon);
  });

  it("WarpPipeline + FinalStage: exportState → importState で出力が再現される", async () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim });
    const pipeline = new WarpPipeline(dim)
      .addIntent({
        tech: { matrix: testMatrix(dim, 1), bias: testVector(dim, 2) },
      })
      .setFinalStage("QuantizationAdapter", quantizer);

    const state = pipeline.exportState();
    const restored = WarpPipeline.importState(state);

    const input = testVector(dim, 44);
    const original = await pipeline.run(input, { intent: "tech" });
    const roundtripped = await restored.run(input, { intent: "tech" });

    expect(original.length).toBe(roundtripped.length);
    expect(maxAbsDiff(original, roundtripped)).toBe(0);
  });
});

describe("プロパティベーステスト: Numerical Stability", () => {
  const dim = 16;

  it("ゼロベクトル入力でクラッシュしない", () => {
    const adapter = new IntentAdapter(dim);
    adapter.addIntent("test", {
      matrix: testMatrix(dim),
      bias: testVector(dim, 1),
    });

    const zeroVec = new Float32Array(dim); // 全てゼロ
    const result = adapter.tune(zeroVec, "test");
    expect(result.length).toBe(dim);
    // NaN が含まれないこと
    for (let i = 0; i < dim; i++) {
      expect(Number.isNaN(result[i])).toBe(false);
    }
  });

  it("非常に大きな値のベクトルでクラッシュしない", () => {
    const adapter = new IntentAdapter(dim);
    adapter.addIntent("test", {
      matrix: testMatrix(dim),
      bias: testVector(dim, 1),
    });

    const largeVec = new Float32Array(dim);
    for (let i = 0; i < dim; i++) largeVec[i] = 1e6;

    const result = adapter.tune(largeVec, "test");
    expect(result.length).toBe(dim);
    // Infinity が含まれないこと（Float32の範囲内であるべき）
    for (let i = 0; i < dim; i++) {
      expect(Number.isFinite(result[i])).toBe(true);
    }
  });

  it("非常に小さな値のベクトルでクラッシュしない", () => {
    const adapter = new IntentAdapter(dim);
    adapter.addIntent("test", {
      matrix: testMatrix(dim),
      bias: testVector(dim, 1),
    });

    const tinyVec = new Float32Array(dim);
    for (let i = 0; i < dim; i++) tinyVec[i] = 1e-38;

    const result = adapter.tune(tinyVec, "test");
    expect(result.length).toBe(dim);
    for (let i = 0; i < dim; i++) {
      expect(Number.isNaN(result[i])).toBe(false);
    }
  });

  it("Int8量子化: ゼロベクトルでクラッシュしない", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim });
    const zeroVec = new Float32Array(dim);
    const result = quantizer.encode(zeroVec);
    expect(result.length).toBeGreaterThan(0);
  });

  it("Binary量子化: ゼロベクトルでクラッシュしない", () => {
    const quantizer = new QuantizationAdapter({ type: "binary", dim });
    const zeroVec = new Float32Array(dim);
    const result = quantizer.encode(zeroVec);
    expect(result.length).toBe(dim / 8);
  });

  it("Int8量子化: 出力値が [-128, 127] の範囲内", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim });

    // 複数のランダムベクトルでテスト
    for (let seed = 0; seed < 20; seed++) {
      const vec = testVector(dim, seed);
      const result = quantizer.encode(vec);
      for (let i = 0; i < result.length; i++) {
        // Int8Array は自動的に [-128, 127] に丸められるが、念のためチェック
        if (result instanceof Int8Array) {
          expect(result[i]).toBeGreaterThanOrEqual(-128);
          expect(result[i]).toBeLessThanOrEqual(127);
        }
      }
    }
  });

  it("パイプライン: 空のパイプラインは入力をそのまま返す", async () => {
    const pipeline = new WarpPipeline(dim);
    const input = testVector(dim, 42);
    const output = await pipeline.run(input);
    expect(output).toBeInstanceOf(Float32Array);
    expect(maxAbsDiff(output as Float32Array, input)).toBe(0);
  });

  it("パイプライン: 同じ入力に対して決定的な出力を返す", async () => {
    const pipeline = new WarpPipeline(dim).addIntent({
      test: { matrix: testMatrix(dim), bias: testVector(dim, 1) },
    });

    const input = testVector(dim, 42);
    const output1 = await pipeline.run(input, { intent: "test" });
    const output2 = await pipeline.run(input, { intent: "test" });

    expect(maxAbsDiff(output1 as Float32Array, output2 as Float32Array)).toBe(
      0,
    );
  });
});
