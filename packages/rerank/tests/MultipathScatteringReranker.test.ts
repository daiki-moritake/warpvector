import { expect, test, describe } from "bun:test";
import { MultipathScatteringReranker } from "../src/rerankers/MultipathScatteringReranker";

describe("MultipathScatteringReranker", () => {
  test("should initialize correctly", () => {
    const reranker = new MultipathScatteringReranker({
      alpha: 0.8,
      threshold: 0.1,
    });
    expect(reranker.alpha).toBe(0.8);
    expect(reranker.threshold).toBe(0.1);
    expect(reranker.maxIterations).toBe(20);
  });

  test("should throw on invalid parameters", () => {
    expect(() => new MultipathScatteringReranker({ alpha: 1.5 })).toThrow();
    expect(() => new MultipathScatteringReranker({ alpha: -0.1 })).toThrow();
    expect(
      () => new MultipathScatteringReranker({ maxIterations: 0 }),
    ).toThrow();
  });

  test("should amplify true hub source based on multipath scattering", () => {
    const reranker = new MultipathScatteringReranker({
      alpha: 0.8,
      threshold: 0.0,
    });

    // A hub document (idx: 1) is similar to both idx: 2 and idx: 3.
    // idx: 0 is an isolated document, but happens to have a high initial score.

    // vectors representing the relationships
    const candidates = [
      new Float32Array([1, 0, 0, 0]), // isolated (doc 0)
      new Float32Array([0, 1, 1, 0]), // hub (doc 1)
      new Float32Array([0, 1, 0, 0]), // connected to hub (doc 2)
      new Float32Array([0, 0, 1, 0]), // connected to hub (doc 3)
    ];

    // Give doc 0 a slightly higher initial score, but it has no multi-path support.
    // Give doc 1 a slightly lower initial score, but it has multi-path support.
    const initialScores = [0.8, 0.7, 0.5, 0.5];

    const results = reranker.rerank(null, candidates, initialScores);

    // Because of multipath scattering, doc 1 (hub) should gather scores from doc 2 and doc 3
    // and potentially surpass doc 0.
    const doc0Result = results.find((r) => r.originalIndex === 0)!;
    const doc1Result = results.find((r) => r.originalIndex === 1)!;

    // The score represents the steady-state probability.
    expect(doc1Result.score).toBeGreaterThan(doc0Result.score);
  });

  test("should preserve original cosine similarities in initialScore when query is provided", () => {
    const reranker = new MultipathScatteringReranker({
      alpha: 0.8,
      threshold: 0.0,
    });

    const query = new Float32Array([1, 0, 0, 0]);
    const candidates = [
      new Float32Array([1, 0, 0, 0]),
      new Float32Array([-1, 0, 0, 0]),
    ];

    const results = reranker.rerank(query, candidates);

    const doc0Result = results.find((r) => r.originalIndex === 0)!;
    const doc1Result = results.find((r) => r.originalIndex === 1)!;

    // query と doc0 の内積は 1.0
    expect(doc0Result.initialScore).toBeCloseTo(1.0, 5);
    // query と doc1 の内積は -1.0
    expect(doc1Result.initialScore).toBeCloseTo(-1.0, 5);
  });

  test("should handle graph with isolated nodes gracefully", () => {
    const reranker = new MultipathScatteringReranker({
      alpha: 0.9,
      threshold: 0.5,
    });

    const candidates = [new Float32Array([1, 0]), new Float32Array([0, 1])];
    // They are completely orthogonal, so threshold=0.5 makes them isolated.

    const initialScores = [0.6, 0.4];
    const results = reranker.rerank(null, candidates, initialScores);

    const doc0Result = results.find((r) => r.originalIndex === 0)!;
    const doc1Result = results.find((r) => r.originalIndex === 1)!;

    // Relative order should remain the same
    expect(doc0Result.score).toBeGreaterThan(doc1Result.score);
  });
});
