import { describe, expect, test } from "bun:test";
import { IntentAdapter } from "../src/adapters/IntentAdapter";
import { LoraIntentAdapter } from "../src/adapters/LoraIntentAdapter";
import { ProjectionAdapter } from "../src/adapters/ProjectionAdapter";

describe("importState validation", () => {
  // --- IntentAdapter ---
  describe("IntentAdapter.importState", () => {
    test("rejects non-JSON string", () => {
      expect(() => IntentAdapter.importState("not json")).toThrow(
        "Failed to parse JSON",
      );
    });

    test("rejects missing dimension", () => {
      expect(() =>
        IntentAdapter.importState(JSON.stringify({ intents: {} })),
      ).toThrow("dimension");
    });

    test("rejects non-integer dimension", () => {
      expect(() =>
        IntentAdapter.importState(
          JSON.stringify({ dimension: 1.5, intents: {} }),
        ),
      ).toThrow("positive integer");
    });

    test("rejects negative dimension", () => {
      expect(() =>
        IntentAdapter.importState(
          JSON.stringify({ dimension: -1, intents: {} }),
        ),
      ).toThrow("positive integer");
    });

    test("rejects missing intents object", () => {
      expect(() =>
        IntentAdapter.importState(JSON.stringify({ dimension: 3 })),
      ).toThrow("intents");
    });

    test("rejects intent with NaN in matrix", () => {
      expect(() =>
        IntentAdapter.importState(
          JSON.stringify({
            dimension: 2,
            intents: {
              bad: {
                matrix: [1, NaN, 0, 1],
                bias: [0, 0],
              },
            },
          }),
        ),
      ).toThrow("finite number");
    });

    test("accepts valid state and round-trips", () => {
      const adapter = new IntentAdapter(2);
      adapter.addIntent("test", {
        matrix: [
          [1, 0],
          [0, 1],
        ],
        bias: [0, 0],
      });
      const state = adapter.exportState!();
      const restored = IntentAdapter.importState(state as string);
      expect(restored).toBeDefined();
      const result = restored.tune([1, 2], "test");
      expect(Array.from(result)).toEqual([1, 2]);
    });
  });

  // --- LoraIntentAdapter ---
  describe("LoraIntentAdapter.importState", () => {
    test("rejects non-JSON", () => {
      expect(() => LoraIntentAdapter.importState("{bad}")).toThrow(
        "Failed to parse JSON",
      );
    });

    test("rejects missing rank", () => {
      expect(() =>
        LoraIntentAdapter.importState(
          JSON.stringify({ dimension: 3, intents: {} }),
        ),
      ).toThrow("rank");
    });

    test("accepts valid state and round-trips", () => {
      const adapter = new LoraIntentAdapter(3, 2);
      adapter.addIntent("test", {
        matrixA: [
          [1, 0],
          [0, 1],
          [0, 0],
        ],
        matrixB: [
          [1, 0, 0],
          [0, 1, 0],
        ],
        bias: [0, 0, 0],
      });
      const state = adapter.exportState!();
      const restored = LoraIntentAdapter.importState(state as string);
      expect(restored).toBeDefined();
    });
  });

  // --- ProjectionAdapter ---
  describe("ProjectionAdapter.importState", () => {
    test("rejects non-JSON", () => {
      expect(() => ProjectionAdapter.importState("garbage")).toThrow(
        "Failed to parse JSON",
      );
    });

    test("rejects missing outDimension", () => {
      expect(() =>
        ProjectionAdapter.importState(
          JSON.stringify({ inDimension: 3, projections: {} }),
        ),
      ).toThrow("outDimension");
    });

    test("accepts valid state and round-trips", () => {
      const adapter = new ProjectionAdapter(3, 2, {
        default: {
          matrix: [
            [1, 0, 0],
            [0, 1, 0],
          ],
        },
      });
      const state = adapter.exportState!();
      const restored = ProjectionAdapter.importState(state as string);
      expect(restored).toBeDefined();
      const result = restored.tune([1, 2, 3], "default");
      expect(result.length).toBe(2);
    });
  });
});

describe("boundary value tests", () => {
  test("IntentAdapter handles zero vector", () => {
    const adapter = new IntentAdapter(3);
    adapter.addIntent("test", {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      bias: [0, 0, 0],
    });
    const result = adapter.tune([0, 0, 0], "test");
    expect(Array.from(result)).toEqual([0, 0, 0]);
  });

  test("IntentAdapter handles large values without overflow", () => {
    const adapter = new IntentAdapter(2);
    adapter.addIntent("test", {
      matrix: [
        [1, 0],
        [0, 1],
      ],
      bias: [0, 0],
    });
    const result = adapter.tune([1e10, -1e10], "test");
    expect(result[0]).toBeCloseTo(1e10, -3);
    expect(result[1]).toBeCloseTo(-1e10, -3);
  });

  test("ProjectionAdapter handles identity projection", () => {
    const adapter = new ProjectionAdapter(2, 2, {
      default: {
        matrix: [
          [1, 0],
          [0, 1],
        ],
      },
    });
    const result = adapter.tune([3.14, 2.72], "default");
    expect(result[0]).toBeCloseTo(3.14, 4);
    expect(result[1]).toBeCloseTo(2.72, 4);
  });

  test("ProjectionAdapter throws on wrong dimension", () => {
    const adapter = new ProjectionAdapter(3, 2, {
      default: {
        matrix: [
          [1, 0, 0],
          [0, 1, 0],
        ],
      },
    });
    expect(() => adapter.tune([1, 2])).toThrow();
  });

  test("IntentAdapter throws on missing intent", () => {
    const adapter = new IntentAdapter(2);
    expect(() => adapter.tune([1, 2], "nonexistent")).toThrow();
  });
});
