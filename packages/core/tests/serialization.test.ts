import { describe, expect, test } from "bun:test";
import { WhiteningAdapter } from "@warpvector/ml";
import { ProjectionAdapter } from "../src/ProjectionAdapter";
import { MlpAdapter } from "@warpvector/ml";

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
});
