import { describe, expect, test } from "bun:test";
import { WarpPipeline } from "../src/pipeline/WarpPipeline";
import { QuantizationAdapter } from "@warpvector/extras";
import { WhiteningAdapter, MlpAdapter } from "@warpvector/ml";

describe("WarpPipeline", () => {
  test("chains adapters and processes vectors correctly", async () => {
    // Pipeline: Intent -> Projection(to 2 dim) -> Quantization(int8)
    const pipeline = new WarpPipeline(3)
      .addIntent({
        my_intent: {
          matrix: [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
          ],
          bias: [1, 1, 1],
        },
      })
      .addProjection(2, {
        my_intent: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
          ],
        },
      });

    const finalStage = new QuantizationAdapter({ type: "int8", dim: 2 });
    pipeline.setFinalStage("QuantizationAdapter", finalStage);

    const input = [0.5, 1.0, 1.5];

    // Step 1: Intent (tune "my_intent") -> Wx + b = [2*0.5+1, 2*1+1, 2*1.5+1] = [2, 3, 4]
    // Step 2: Projection (1st 2 elements) -> [2, 3]
    // Step 3: Quantize int8 -> clamp and int8 -> Int8Array[2, 3]

    const result = await pipeline.run(input, { intent: "my_intent" });

    expect(result).toBeInstanceOf(Int8Array);
    expect(result.length).toBe(2);
    // [2, 3] は 127 を掛けるとそれぞれ > 127 となるため、127にクリップされる
    expect(result[0]).toBe(127);
    expect(result[1]).toBe(127);
  });

  test("can export and import state completely", () => {
    const pipeline = new WarpPipeline(8)
      .addStep(
        "WhiteningAdapter",
        new WhiteningAdapter(8, { numComponents: 1 }),
      )
      .addIntent({
        test: {
          matrix: [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
          ],
          bias: [0, 0, 0, 0, 0, 0, 0, 0],
        },
      });

    pipeline.setFinalStage(
      "QuantizationAdapter",
      new QuantizationAdapter({ type: "binary", dim: 8 }),
    );

    const state = pipeline.exportState();
    expect(state.steps.length).toBe(2);
    expect(state.steps[0].type).toBe("WhiteningAdapter");
    expect(state.steps[1].type).toBe("IntentAdapter");
    expect(state.finalStage?.type).toBe("QuantizationAdapter");

    const restoredPipeline = WarpPipeline.importState(state);

    const restoredState = restoredPipeline.exportState();
    expect(restoredState.steps.length).toBe(2);
    expect(restoredState.steps[0].type).toBe("WhiteningAdapter");
    expect(restoredState.finalStage?.type).toBe("QuantizationAdapter");
  });

  test("throws error when importing empty states", () => {
    expect(() => WarpPipeline.importState([])).toThrow();
  });

  test("throws error when importing state with unregistered final stage", () => {
    const state = {
      steps: [
        {
          type: "IntentAdapter",
          state: JSON.stringify({
            dimension: 2,
            intents: {
              test: {
                matrix: [1, 0, 0, 1],
                bias: [0, 0],
              },
            },
          }),
        },
      ],
      finalStage: {
        type: "NonExistentFinalStage",
        state: "{}",
      },
    };

    expect(() => WarpPipeline.importState(state)).toThrow(
      "Unknown final stage adapter type: NonExistentFinalStage",
    );
  });

  test("runBatch processes multiple vectors correctly", async () => {
    const pipeline = new WarpPipeline(3).addIntent({
      my_intent: {
        matrix: [
          [2, 0, 0],
          [0, 2, 0],
          [0, 0, 2],
        ],
        bias: [0, 0, 0],
      },
    });

    const batch = [
      [1, 2, 3],
      [4, 5, 6],
    ];

    const result = await pipeline.runBatch(batch, { intent: "my_intent" });
    expect(result.length).toBe(2);
    expect(Array.from(result[0])).toEqual([2, 4, 6]);
    expect(Array.from(result[1])).toEqual([8, 10, 12]);
  });

  test("runAndFormat outputs to pgvector format correctly", async () => {
    const pipeline = new WarpPipeline(2); // no-op pipeline for testing formatting
    const result = await pipeline.runAndFormat(new Float32Array([0.1, 0.2]), {
      format: "pgvector",
    });
    expect(typeof result).toBe("string");
    // Float32の精度で値が格納されるため、文字列表現が若干異なる場合があるが、
    // toPgvectorの形式（[x, y]）であることを確認
    expect((result as string).startsWith("[")).toBe(true);
    expect((result as string).endsWith("]")).toBe(true);
  });

  test("runAndFormat outputs to pinecone format correctly", async () => {
    const pipeline = new WarpPipeline(2);
    const result = (await pipeline.runAndFormat(new Float32Array([0.5, 1.0]), {
      format: "pinecone",
      topK: 5,
      filter: { genre: "action" },
    })) as { vector: number[]; topK: number; filter: Record<string, unknown> };
    expect(result.vector.length).toBe(2);
    expect(result.topK).toBe(5);
    expect(result.filter.genre).toBe("action");
  });

  test("runAndFormat outputs to redis format correctly", async () => {
    const pipeline = new WarpPipeline(2);
    const result = (await pipeline.runAndFormat([1.0, -1.0], {
      format: "redis",
    })) as Uint8Array;
    expect(result).toBeInstanceOf(Uint8Array);
    // Float32Array [1.0, -1.0] has byte length 8
    expect(result.length).toBe(8);
  });

  test("init successfully initializes async adapters", async () => {
    const mlpAdapter = new MlpAdapter([
      {
        matrix: [
          [1, 0],
          [0, 1],
        ],
        bias: [0, 0],
        activation: "linear",
      },
    ]);

    const pipeline = new WarpPipeline(2).addStep("MlpAdapter", mlpAdapter);

    await pipeline.init();

    // Test if run works after init
    const result = await pipeline.run([1, 2]);
    expect(Array.from(result)).toEqual([1, 2]);
  });

  test("supports custom adapters via AdapterRegistry", async () => {
    // カスタムアダプタの定義
    class MyCustomAdapter {
      constructor(public scale: number) {}

      public tune(vector: number[] | Float32Array): Float32Array {
        return new Float32Array(Array.from(vector).map((v) => v * this.scale));
      }

      public exportState() {
        return { scale: this.scale };
      }

      public static importState(state: any) {
        return new MyCustomAdapter(state.scale);
      }
    }

    // レジストリに登録
    WarpPipeline.registerAdapter(
      "MyCustomAdapter",
      MyCustomAdapter.importState,
    );

    // パイプラインを直接構築してカスタムアダプタを追加
    const pipeline = new WarpPipeline(3).addStep(
      "MyCustomAdapter",
      new MyCustomAdapter(5),
    );

    // カスタムアダプタが正しく実行されるか確認
    const result = await pipeline.run([1, 2, 3]);
    expect(Array.from(result)).toEqual([5, 10, 15]);

    // さらに状態を正しくエクスポートできるか確認
    const exportedState = pipeline.exportState();
    expect(exportedState.steps.length).toBe(1);
    expect(exportedState.steps[0].type).toBe("MyCustomAdapter");
    expect(exportedState.steps[0].state).toEqual({ scale: 5 });

    // エクスポートした状態から完全に復元できるか確認
    const restoredPipeline = WarpPipeline.importState(exportedState);
    const result2 = await restoredPipeline.run([2, 4, 6]);
    expect(Array.from(result2)).toEqual([10, 20, 30]);
  });
});
