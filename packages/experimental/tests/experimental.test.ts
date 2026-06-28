/**
 * @warpvector/experimental パッケージのテスト
 *
 * 実験的機能のre-exportが正しく動作することを検証
 */
import { describe, test, expect } from "bun:test";
import {
  ColbertAdapter,
  VsaAdapter,
  AnomalyDetectionAdapter,
  TaskArithmetic,
} from "@warpvector/experimental";

describe("@warpvector/experimental", () => {
  test("ColbertAdapter がインポートできる", () => {
    expect(ColbertAdapter).toBeDefined();
    expect(typeof ColbertAdapter).toBe("function");
  });

  test("VsaAdapter がインポートできる", () => {
    expect(VsaAdapter).toBeDefined();
    expect(typeof VsaAdapter).toBe("function");
  });

  test("AnomalyDetectionAdapter がインポートできる", () => {
    expect(AnomalyDetectionAdapter).toBeDefined();
    expect(typeof AnomalyDetectionAdapter).toBe("function");
  });

  test("TaskArithmetic がインポートできる", () => {
    expect(TaskArithmetic).toBeDefined();
    expect(typeof TaskArithmetic).toBe("function");
  });

  test("VsaAdapter.bundle() で複数ベクトルを結合できる", () => {
    const v1 = new Float32Array(64).fill(1);
    const v2 = new Float32Array(64).fill(-1);
    const bundled = VsaAdapter.bundle([v1, v2]);
    expect(bundled).toBeInstanceOf(Float32Array);
    expect(bundled.length).toBe(64);
  });

  test("AnomalyDetectionAdapter で異常検知が動作する", () => {
    const detector = new AnomalyDetectionAdapter({
      mode: "safe",
      maxValue: 1.0,
    });
    // NaN を含むベクトル → safe mode でゼロ埋め
    const input = new Float32Array([0.5, NaN, 0.3, 999]);
    const result = detector.tune(input);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result[1]).toBe(0); // NaN がゼロになる
    expect(result[3]).toBe(1.0); // maxValue にクリップ
  });
});
