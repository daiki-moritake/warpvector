import { expect, test, describe } from "bun:test";
import { AnomalyDetectionAdapter } from "../src/adapters/AnomalyDetectionAdapter";
import { SafeQuantizationAdapter } from "../src/adapters/SafeQuantizationAdapter";

describe("AnomalyDetectionAdapter", () => {
  test("strict mode throws error on NaN, Infinity, and OutOfBounds", () => {
    const strictAdapter = new AnomalyDetectionAdapter({ mode: "strict", maxValue: 100.0 });
    
    // Normal vector
    expect(() => strictAdapter.tune([1, 2, 3])).not.toThrow();

    // NaN
    expect(() => strictAdapter.tune([1, NaN, 3])).toThrow("AnomalyDetectionAdapter [STRICT MODE]: NaN detected");

    // Infinity
    expect(() => strictAdapter.tune([1, Infinity, 3])).toThrow("AnomalyDetectionAdapter [STRICT MODE]: Infinity detected");

    // MaxValue exceeded
    expect(() => strictAdapter.tune([100.1, 2, 3])).toThrow("AnomalyDetectionAdapter [STRICT MODE]: Value 100.1 exceeds maxValue");
  });

  test("safe mode clips and sanitizes vectors", () => {
    const safeAdapter = new AnomalyDetectionAdapter({ mode: "safe", maxValue: 50.0 });

    const input = [1, NaN, Infinity, 100, -60, 25];
    const output = safeAdapter.tune(input);

    expect(output[0]).toBe(1);
    expect(output[1]).toBe(0); // NaN -> 0
    expect(output[2]).toBe(0); // Infinity -> 0
    expect(output[3]).toBe(50); // 100 -> 50 (clipped)
    expect(output[4]).toBe(-50); // -60 -> -50 (clipped)
    expect(output[5]).toBe(25);
  });
});

describe("SafeQuantizationAdapter", () => {
  test("wraps QuantizationAdapter and prevents overflow for int8", () => {
    // QuantizationAdapterは、non-dynamicのint8の場合、入力を[-1.0, 1.0]と仮定して127倍します。
    // そのため、clipThresholdを1.0に設定して安全を担保します。
    const safeQ = new SafeQuantizationAdapter({ type: "int8", dim: 3, clipThreshold: 1.0 });

    const input = [0.1, 2.0, -3.0]; 
    // 0.1 * 127 = 13 (Math.round)
    // 2.0 > 1.0 なので 1.0にクリップ -> 127
    // -3.0 < -1.0 なので -1.0にクリップ -> -127 (QuantizationAdapterでは -128)
    
    const output = safeQ.tune(input) as Int8Array;

    expect(output.length).toBe(3);
    expect(output[0]).toBe(13);
    expect(output[1]).toBe(127);
    expect(output[2]).toBe(-127); // -1.0 * 127 = -127
  });

  test("handles NaN safely", () => {
    const safeQ = new SafeQuantizationAdapter({ type: "int8", dim: 2, clipThreshold: 1.0 });
    const output = safeQ.tune([0.1, NaN]) as Int8Array;
    expect(output[0]).toBe(13);
    expect(output[1]).toBe(0); // NaN -> 0
  });
});
