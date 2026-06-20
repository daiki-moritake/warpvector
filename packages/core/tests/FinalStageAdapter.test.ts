import { describe, expect, test } from "bun:test";
import { WarpPipeline } from "../src/pipeline/WarpPipeline";
import { QuantizationAdapter } from "@warpvector/extras";

describe("FinalStageAdapter pipeline", () => {
  test("setFinalStage applies quantization at the end of pipeline", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim: 3 });

    const pipeline = new WarpPipeline(3)
      .addIntent({
        my_intent: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
          bias: [0, 0, 0],
        },
      })
      .setFinalStage("QuantizationAdapter", quantizer);

    const result = pipeline.run([0.5, -0.3, 0.8], { intent: "my_intent" });

    // FinalStage 経由なので Int8Array が返る
    expect(result).toBeInstanceOf(Int8Array);
    expect(result.length).toBe(3);
  });

  test("pipeline without finalStage returns Float32Array", () => {
    const pipeline = new WarpPipeline(3).addIntent({
      default: {
        matrix: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        bias: [0, 0, 0],
      },
    });

    const result = pipeline.run([1, 2, 3], { intent: "default" });
    expect(result).toBeInstanceOf(Float32Array);
  });

  test("runBatch with finalStage quantizes all vectors", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim: 2 });

    const pipeline = new WarpPipeline(2)
      .addIntent({
        test: {
          matrix: [
            [1, 0],
            [0, 1],
          ],
          bias: [0, 0],
        },
      })
      .setFinalStage("QuantizationAdapter", quantizer);

    const results = pipeline.runBatch(
      [
        [0.5, -0.5],
        [1.0, 0.0],
      ],
      { intent: "test" },
    );

    expect(results.length).toBe(2);
    for (const r of results) {
      expect(r).toBeInstanceOf(Int8Array);
      expect(r.length).toBe(2);
    }
  });

  test("exportState includes finalStage", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim: 3 });

    const pipeline = new WarpPipeline(3)
      .addIntent({
        default: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
          bias: [0, 0, 0],
        },
      })
      .setFinalStage("QuantizationAdapter", quantizer);

    const state = pipeline.exportState();
    expect(state.steps.length).toBe(1);
    expect(state.steps[0].type).toBe("IntentAdapter");
    expect(state.finalStage).toBeDefined();
    expect(state.finalStage!.type).toBe("QuantizationAdapter");
    expect(state.finalStage!.state).toBeDefined();
  });

  test("importState restores finalStage", () => {
    const quantizer = new QuantizationAdapter({ type: "int8", dim: 3 });

    const pipeline = new WarpPipeline(3)
      .addIntent({
        default: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
          bias: [0, 0, 0],
        },
      })
      .setFinalStage("QuantizationAdapter", quantizer);

    const state = pipeline.exportState();
    const restored = WarpPipeline.importState(state);
    const restoredState = restored.exportState();

    expect(restoredState.steps.length).toBe(1);
    expect(restoredState.finalStage).toBeDefined();
    expect(restoredState.finalStage!.type).toBe("QuantizationAdapter");
  });

  test("backward compatible: importState accepts legacy flat array", () => {
    // 旧形式: PipelineState[] をそのまま渡す
    const legacyState = [
      {
        type: "IntentAdapter",
        state: JSON.stringify({
          dimension: 2,
          intents: {
            default: {
              matrix: [1, 0, 0, 1],
              bias: [0, 0],
            },
          },
        }),
      },
    ];

    const restored = WarpPipeline.importState(legacyState);
    const result = restored.run([3, 4], { intent: "default" });
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(2);
  });
});
