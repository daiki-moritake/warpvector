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
    // safe mode では外れ値が全体スケーリングされる
    const input = new Float32Array([0.5, 0.2, 0.3, 2.0]);
    const result = detector.tune(input);
    expect(result).toBeInstanceOf(Float32Array);
    // maxValue = 1.0, maxAbs = 2.0 -> scale = 0.5
    expect(result[3]).toBe(1.0); 
    expect(result[0]).toBe(0.25); 
    
    // NaN は safe mode でもエラーになる
    const nanInput = new Float32Array([0.5, NaN, 0.3, 999]);
    expect(() => detector.tune(nanInput)).toThrow(/Invalid value \(NaN\)/);
  });
});
