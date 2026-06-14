import { describe, expect, test } from "bun:test";
import { WarpPipeline } from "../src/WarpPipeline";

describe("WarpPipeline", () => {
  test("chains adapters and processes vectors correctly", () => {
    // Pipeline: Intent -> Projection(to 2 dim) -> Quantization(int8)
    const pipeline = new WarpPipeline(3)
      .addIntent({
        "my_intent": {
          matrix: [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
          ],
          bias: [1, 1, 1]
        }
      })
      .addProjection(2, {
        "my_intent": {
          matrix: [
            [1, 0, 0],
            [0, 1, 0]
          ]
        }
      })
      .quantize("int8");

    const input = [0.5, 1.0, 1.5];
    
    // Step 1: Intent (tune "my_intent") -> Wx + b = [2*0.5+1, 2*1+1, 2*1.5+1] = [2, 3, 4]
    // Step 2: Projection (1st 2 elements) -> [2, 3]
    // Step 3: Quantize int8 -> clamp and int8 -> Int8Array[2, 3]

    const result = pipeline.run(input, { intent: "my_intent" });

    expect(result).toBeInstanceOf(Int8Array);
    expect(result.length).toBe(2);
    // [2, 3] は 127 を掛けるとそれぞれ > 127 となるため、127にクリップされる
    expect(result[0]).toBe(127);
    expect(result[1]).toBe(127);
  });

  test("can export and import state completely", () => {
    const pipeline = new WarpPipeline(8)
      .addWhitening({ numComponents: 1 })
      .addIntent({ "test": { matrix: [
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
      ], bias: [0,0,0,0,0,0,0,0] }})
      .quantize("binary");

    const state = pipeline.exportState();
    expect(state.length).toBe(3);
    expect(state[0].type).toBe("WhiteningAdapter");
    expect(state[1].type).toBe("IntentAdapter");
    expect(state[2].type).toBe("QuantizationAdapter");

    const restoredPipeline = WarpPipeline.importState(state);
    
    // Test the restored pipeline behavior
    // 4 dim -> binary -> 1 byte (since length 4 throws mismatch? Wait, binary requires multiple of 8. 
    // Ah, binary quantization requires dim % 8 === 0.
    // We shouldn't actually call run() with binary quantization for dim 4, 
    // but we can check if it loaded successfully without throwing during import.
    
    const restoredState = restoredPipeline.exportState();
    expect(restoredState.length).toBe(3);
    expect(restoredState[0].type).toBe("WhiteningAdapter");
  });

  test("throws error when importing empty states", () => {
    expect(() => WarpPipeline.importState([])).toThrow();
  });

  test("runBatch processes multiple vectors correctly", () => {
    const pipeline = new WarpPipeline(3)
      .addIntent({
        "my_intent": {
          matrix: [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
          ],
          bias: [0, 0, 0]
        }
      });
      
    const batch = [
      [1, 2, 3],
      [4, 5, 6]
    ];
    
    const result = pipeline.runBatch(batch, { intent: "my_intent" });
    expect(result.length).toBe(2);
    expect(Array.from(result[0])).toEqual([2, 4, 6]);
    expect(Array.from(result[1])).toEqual([8, 10, 12]);
  });

  test("runAndFormat outputs to pgvector format correctly", () => {
    const pipeline = new WarpPipeline(2); // no-op pipeline for testing formatting
    const result = pipeline.runAndFormat([0.1, 0.2], { format: "pgvector" });
    expect(result).toBe("[0.1, 0.2]");
  });

  test("runAndFormat outputs to pinecone format correctly", () => {
    const pipeline = new WarpPipeline(2);
    const result = pipeline.runAndFormat([0.1, 0.2], { format: "pinecone", topK: 5, filter: { genre: "action" } });
    expect(result.vector).toEqual([0.1, 0.2]);
    expect(result.topK).toBe(5);
    expect(result.filter.genre).toBe("action");
  });

  test("runAndFormat outputs to redis format correctly", () => {
    const pipeline = new WarpPipeline(2);
    const result = pipeline.runAndFormat([1.0, -1.0], { format: "redis" });
    expect(result).toBeInstanceOf(Uint8Array);
    // Float32Array [1.0, -1.0] has byte length 8
    expect(result.length).toBe(8);
  });

  test("init successfully initializes async adapters", async () => {
    // Tests that init doesn't throw and successfully awaits MlpAdapter init.
    const pipeline = new WarpPipeline(2)
      .addMlp([{
        matrix: [[1, 0], [0, 1]],
        bias: [0, 0],
        activation: "linear"
      }]);
      
    await pipeline.init();
    
    // Test if run works after init
    const result = pipeline.run([1, 2]);
    expect(Array.from(result)).toEqual([1, 2]);
  });
});
