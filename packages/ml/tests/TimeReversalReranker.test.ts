import { expect, test, describe } from "bun:test";
import { TimeReversalReranker } from "../src/rerankers/TimeReversalReranker";
import { normalize, innerProduct } from "@warpvector/core";

describe("TimeReversalReranker", () => {
  test("should initialize correctly", () => {
    const reranker = new TimeReversalReranker({ tau: 2.5, threshold: 0.1, normalizeGraph: false, iterations: 2 });
    expect(reranker.tau).toBe(2.5);
    expect(reranker.threshold).toBe(0.1);
    expect(reranker.normalizeGraph).toBe(false);
    expect(reranker.iterations).toBe(2);
  });

  test("should throw on invalid parameters", () => {
    expect(() => new TimeReversalReranker({ tau: -1.0 })).toThrow("non-negative");
    expect(() => new TimeReversalReranker({ iterations: 0 })).toThrow("at least 1");
  });

  test("should amplify true source and suppress diffused neighbors", () => {
    const reranker = new TimeReversalReranker({ tau: 1.0, threshold: 0.0, normalizeGraph: false });
    
    const source = normalize(new Float32Array([1.0, 0.0, 0.0]));
    const neighbor1 = normalize(new Float32Array([0.8, 0.6, 0.0]));
    const neighbor2 = normalize(new Float32Array([0.8, -0.6, 0.0]));
    const independent = normalize(new Float32Array([0.0, 0.0, 1.0]));
    const query = normalize(new Float32Array([1.0, 0.1, 0.0]));

    const candidates = [source, neighbor1, neighbor2, independent];

    const s0_source = innerProduct(query, source);
    const s0_n1 = innerProduct(query, neighbor1);

    expect(s0_source).toBeCloseTo(1.0, 1);
    expect(s0_n1).toBeGreaterThan(0.7);

    const results = reranker.rerank(query, candidates);

    const res_source = results.find(r => r.originalIndex === 0)!;
    const res_n1 = results.find(r => r.originalIndex === 1)!;
    const res_n2 = results.find(r => r.originalIndex === 2)!;
    const res_ind = results.find(r => r.originalIndex === 3)!;

    expect(res_source.score).toBeGreaterThan(res_source.initialScore);
    expect(res_n1.score).toBeLessThan(res_n1.initialScore);
    expect(res_n2.score).toBeLessThan(res_n2.initialScore);

    const initialGap = res_source.initialScore - res_n1.initialScore;
    const rerankedGap = res_source.score - res_n1.score;
    expect(rerankedGap).toBeGreaterThan(initialGap);
    expect(res_ind.score).toBeCloseTo(res_ind.initialScore, 5);
  });

  test("should handle initialScores without query", () => {
    const reranker = new TimeReversalReranker({ tau: 1.0, threshold: 0.0 });
    
    const source = normalize(new Float32Array([1.0, 0.0, 0.0]));
    const neighbor1 = normalize(new Float32Array([0.8, 0.6, 0.0]));
    const candidates = [source, neighbor1];
    const initialScores = [0.99, 0.85];

    // No query provided!
    const results = reranker.rerank(null, candidates, initialScores);

    const res_source = results.find(r => r.originalIndex === 0)!;
    const res_n1 = results.find(r => r.originalIndex === 1)!;

    expect(res_source.initialScore).toBeCloseTo(0.99, 5);
    expect(res_n1.initialScore).toBeCloseTo(0.85, 5);

    // Source should pull energy from neighbor
    expect(res_source.score).toBeGreaterThan(0.99);
    expect(res_n1.score).toBeLessThan(0.85);
  });

  test("should apply graph normalization to prevent hub explosion", () => {
    // Unnormalized vs Normalized
    const rerankerUnnorm = new TimeReversalReranker({ tau: 1.0, threshold: 0.0, normalizeGraph: false });
    const rerankerNorm = new TimeReversalReranker({ tau: 1.0, threshold: 0.0, normalizeGraph: true });
    
    // Hub is connected to many, but has slightly lower initial score than source
    const hub = normalize(new Float32Array([1.0, 0.0, 0.0])); 
    const source = normalize(new Float32Array([1.0, 0.0, 0.0])); // They are essentially the same for distance matrix
    
    // Create 10 neighbors identical to Hub to simulate a massive cluster
    const candidates = [hub, source];
    for (let i = 0; i < 10; i++) {
        candidates.push(normalize(new Float32Array([1.0, 0.0, 0.0])));
    }
    
    // Let's artificially give Hub a slightly higher initial score than its neighbors, 
    // and Source an even higher one.
    const initialScores = [0.9, 0.95]; // Hub: 0.9, Source: 0.95
    for (let i = 0; i < 10; i++) {
        initialScores.push(0.8); // Neighbors: 0.8
    }

    const resUnnorm = rerankerUnnorm.rerank(null, candidates, initialScores);
    const resNorm = rerankerNorm.rerank(null, candidates, initialScores);

    const hubUnnorm = resUnnorm.find(r => r.originalIndex === 0)!;
    const hubNorm = resNorm.find(r => r.originalIndex === 0)!;

    // Unnormalized hub score will explode because it pulls (0.9 - 0.8) from 10 neighbors = +1.0!
    // (Wait, actually it pulls tau * W * (0.9 - 0.8) * 10 = 1.0 * 1.0 * 0.1 * 10 = 1.0)
    // So 0.9 + 1.0 = 1.9
    expect(hubUnnorm.score).toBeGreaterThan(1.8);

    // Normalized hub score will divide the sum by Degree (which is ~11)
    // So 0.9 + (1.0 / 11) = ~0.99
    // It remains bounded!
    expect(hubNorm.score).toBeLessThan(1.5);
    expect(hubNorm.score).toBeGreaterThan(0.9);
  });

  test("should handle multiple iterations safely", () => {
    const reranker1 = new TimeReversalReranker({ tau: 1.0, iterations: 1 });
    const reranker3 = new TimeReversalReranker({ tau: 0.333, iterations: 3 });

    const source = normalize(new Float32Array([1.0, 0.0, 0.0]));
    const neighbor1 = normalize(new Float32Array([0.8, 0.6, 0.0]));
    const candidates = [source, neighbor1];
    const initialScores = [0.9, 0.8];

    const res1 = reranker1.rerank(null, candidates, initialScores);
    const res3 = reranker3.rerank(null, candidates, initialScores);

    const source1 = res1.find(r => r.originalIndex === 0)!;
    const source3 = res3.find(r => r.originalIndex === 0)!;

    // Both should amplify the source
    expect(source1.score).toBeGreaterThan(0.9);
    expect(source3.score).toBeGreaterThan(0.9);
    // Multiple smaller iterations should yield roughly similar but more stable results
    expect(Math.abs(source1.score - source3.score)).toBeLessThan(0.15);
  });

  test("performance test for graph construction", () => {
    const reranker = new TimeReversalReranker({ tau: 1.0, threshold: 0.1 });
    const N = 500;
    const dim = 128; // Standard embed size
    
    // Generate 500 random vectors
    const candidates: Float32Array[] = [];
    for (let i = 0; i < N; i++) {
        const vec = new Float32Array(dim);
        for (let j = 0; j < dim; j++) {
            vec[j] = Math.random() - 0.5;
        }
        candidates.push(normalize(vec));
    }
    const query = candidates[0]; // Query is the first candidate

    const start = performance.now();
    reranker.rerank(query, candidates);
    const end = performance.now();
    
    const durationMs = end - start;
    // We expect 500x500 pairwise distance (125,000 comparisons of 128 dim) 
    // to take less than 50ms in Node.js thanks to our optimization
    expect(durationMs).toBeLessThan(100); 
  });
});
