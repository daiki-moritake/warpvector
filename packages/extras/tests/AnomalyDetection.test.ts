import { expect, test, describe } from "bun:test";
import { AnomalyDetectionAdapter } from "../src/adapters/AnomalyDetectionAdapter";
import { SafeQuantizationAdapter } from "../src/adapters/SafeQuantizationAdapter";
import "../src/index";

describe("AnomalyDetectionAdapter", () => {
  test("strict mode throws error on NaN, Infinity, and OutOfBounds", () => {
    const strictAdapter = new AnomalyDetectionAdapter({
      mode: "strict",
      maxValue: 100.0,
    });

    // Normal vector
    expect(() => strictAdapter.tune([1, 2, 3])).not.toThrow();

    // NaN
    expect(() => strictAdapter.tune([1, NaN, 3])).toThrow(
      "AnomalyDetectionAdapter [STRICT MODE]: NaN detected",
    );

    // Infinity
    expect(() => strictAdapter.tune([1, Infinity, 3])).toThrow(
      "AnomalyDetectionAdapter [STRICT MODE]: Infinity detected",
    );

    // MaxValue exceeded
    expect(() => strictAdapter.tune([100.1, 2, 3])).toThrow(
      "AnomalyDetectionAdapter [STRICT MODE]: Value 100.1 exceeds maxValue",
    );
  });

  test("rejects invalid maxValue values", () => {
    expect(() => {
      new AnomalyDetectionAdapter({
        maxValue: -5.0,
      });
    }).toThrow("maxValue");

    expect(() => {
      new AnomalyDetectionAdapter({
        maxValue: 0,
      });
    }).toThrow("maxValue");

    expect(() => {
      new AnomalyDetectionAdapter({
        maxValue: NaN,
      });
    }).toThrow("maxValue");
  });

  test("safe mode scales vector uniformly to preserve direction", () => {
    const safeAdapter = new AnomalyDetectionAdapter({
      mode: "safe",
      maxValue: 50.0,
    });

    // maxAbs is |-100| = 100
    // scale = 50.0 / 100 = 0.5
    const input = [10, 100, -60, 25];
    const output = safeAdapter.tune(input);

    expect(output[0]).toBe(5);   // 10 * 0.5
    expect(output[1]).toBe(50);  // 100 * 0.5
    expect(output[2]).toBe(-30); // -60 * 0.5
    expect(output[3]).toBe(12.5);// 25 * 0.5
  });

  test("safe mode throws error on NaN and Infinity", () => {
    const safeAdapter = new AnomalyDetectionAdapter({
      mode: "safe",
      maxValue: 50.0,
    });

    expect(() => safeAdapter.tune([1, NaN, 3])).toThrow("Invalid value (NaN) detected");
    expect(() => safeAdapter.tune([1, Infinity, 3])).toThrow("Invalid value (Infinity) detected");
  });
});

describe("SafeQuantizationAdapter", () => {
  test("wraps QuantizationAdapter and prevents overflow for int8 by scaling uniformly", () => {
    // QuantizationAdapterは、non-dynamicのint8の場合、入力を[-1.0, 1.0]と仮定して127倍します。
    // そのため、clipThresholdを1.0に設定して安全を担保します。
    const safeQ = new SafeQuantizationAdapter({
      type: "int8",
      dim: 3,
      clipThreshold: 1.0,
    });

    const input = [0.1, 2.0, -3.0];
    // max absolute value is |-3.0| = 3.0
    // scale = 1.0 / 3.0
    // 0.1 * (1/3) * 127 = 4 (Math.round)
    // 2.0 * (1/3) * 127 = 85 (Math.round)
    // -3.0 * (1/3) * 127 = -127

    const output = safeQ.encode(new Float32Array(input)) as Int8Array;

    expect(output.length).toBe(3);
    expect(output[0]).toBe(4);
    expect(output[1]).toBe(85);
    expect(output[2]).toBe(-127);
  });

  test("throws error when NaN is encountered", () => {
    const safeQ = new SafeQuantizationAdapter({
      type: "int8",
      dim: 2,
      clipThreshold: 1.0,
    });
    expect(() => {
      safeQ.encode(new Float32Array([0.1, NaN]));
    }).toThrow("encountered an invalid value (NaN)");
  });

  test("WarpPipeline serialization and deserialization works", async () => {
    const { WarpPipeline } = require("@warpvector/core");
    const pipeline = new WarpPipeline(3).addStep(
      "AnomalyDetectionAdapter",
      new AnomalyDetectionAdapter({ mode: "safe", maxValue: 50.0 }),
    );
    pipeline.setFinalStage(
      "SafeQuantizationAdapter",
      new SafeQuantizationAdapter({ type: "int8", dim: 3, clipThreshold: 1.0 }),
    );

    const state = pipeline.exportState();
    const restored = WarpPipeline.importState(state);

    expect(restored).toBeDefined();

    // 復元されたパイプラインを実行して、元の動作と同じか確認
    const input = [0.1, 2.0, -3.0];
    const output = (await restored.run(input)) as Int8Array;

    expect(output.length).toBe(3);
    expect(output[0]).toBe(4);
    expect(output[1]).toBe(85);
    expect(output[2]).toBe(-127);
  });

  test("rejects invalid clipThreshold values", () => {
    expect(() => {
      new SafeQuantizationAdapter({
        type: "int8",
        dim: 3,
        clipThreshold: -1.0,
      });
    }).toThrow("clipThreshold");

    expect(() => {
      new SafeQuantizationAdapter({
        type: "int8",
        dim: 3,
        clipThreshold: 0,
      });
    }).toThrow("clipThreshold");

    expect(() => {
      new SafeQuantizationAdapter({
        type: "int8",
        dim: 3,
        clipThreshold: NaN,
      });
    }).toThrow("clipThreshold");
  });
});
