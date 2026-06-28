import { describe, expect, test } from "bun:test";
import { WhiteningAdapter } from "@warpvector/ml";
import { ProjectionAdapter } from "../src/adapters/ProjectionAdapter";
import { MlpAdapter } from "@warpvector/ml";
import { WarpPipeline } from "../src/pipeline/WarpPipeline";

describe("Universal Serialization", () => {
  test("WhiteningAdapter serialization", () => {
    const adapter = new WhiteningAdapter(4, {
      learningRate: 0.05,
      numComponents: 2,
    });

    // Simulate some learning
    adapter.update([1.0, 2.0, 3.0, 4.0]);
    adapter.update([-1.0, -2.0, -3.0, -4.0]);

    const state = adapter.exportState();
    expect(typeof state).toBe("string");

    const restored = WhiteningAdapter.importState(state);

    // Verify properties
    expect(restored.dim).toBe(4);
    expect(restored["learningRate"]).toBe(0.05); // Access private for test check
    expect(restored["numComponents"]).toBe(2);
    expect(restored.mean).toEqual(adapter.mean);
    expect(restored.components.length).toBe(2);
    expect(restored.components[0]).toEqual(adapter.components[0]);

    // Verify output consistency
    const input = [1, 1, 1, 1];
    const originalOutput = adapter.tune(input);
    const restoredOutput = restored.tune(input);
    expect(originalOutput).toEqual(restoredOutput);
  });

  test("ProjectionAdapter serialization", () => {
    const adapter = new ProjectionAdapter(3, 2);
    adapter.addProjection("test", {
      matrix: [
        [1, 2, 3],
        [4, 5, 6],
      ],
      bias: [0.1, 0.2],
    });

    const state = adapter.exportState();
    const restored = ProjectionAdapter.importState(state);

    const input = [1, 2, 3];
    const originalOutput = adapter.tune(input, "test");
    const restoredOutput = restored.tune(input, "test");
    expect(originalOutput).toEqual(restoredOutput);
  });

  test("MlpAdapter serialization", async () => {
    const adapter = new MlpAdapter([
      {
        matrix: [
          [1, 2],
          [3, 4],
        ],
        bias: [0.1, 0.2],
        activation: "relu",
      },
    ]);
    await adapter.init();

    const state = adapter.exportState();
    const restored = MlpAdapter.importState(state);
    await restored.init(); // Required after import

    const input = [1, -1];
    const originalOutput = adapter.tune(input);
    const restoredOutput = restored.tune(input);
    expect(originalOutput).toEqual(restoredOutput);
  });

  test("WarpPipeline with ProjectionAdapter serialization and dryRun", async () => {
    const pipeline = new WarpPipeline(3);
    pipeline.addProjection(2, {
      default: {
        matrix: [
          [1, 0, 0],
          [0, 1, 0],
        ],
        bias: [0.1, 0.2],
      },
    });

    // Check dryRun is executable and runs correctly
    const dryRunRes = await pipeline.dryRun([1, 2, 3]);
    expect(dryRunRes.length).toBe(1);
    expect(dryRunRes[0].step).toBe("ProjectionAdapter");
    const dryRunOut = dryRunRes[0].output as Float32Array;
    expect(dryRunOut[0]).toBeCloseTo(1.1, 5);
    expect(dryRunOut[1]).toBeCloseTo(2.2, 5);

    const state = pipeline.exportState();
    expect(state.steps[0].type).toBe("ProjectionAdapter");

    const restored = WarpPipeline.importState(state);
    expect(restored.inputDim).toBe(3); // Verify dimension restoration

    const input = [2, 3, 4];
    const originalOutput = (await pipeline.run(input)) as Float32Array;
    const restoredOutput = (await restored.run(input)) as Float32Array;

    expect(originalOutput).toEqual(restoredOutput);
    expect(originalOutput[0]).toBeCloseTo(2.1, 5);
    expect(originalOutput[1]).toBeCloseTo(3.2, 5);
  });
});
