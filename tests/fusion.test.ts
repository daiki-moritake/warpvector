import { describe, expect, test } from "bun:test";
import { rrf, rsf, RankedResult } from "../src/fusion";

describe("Reciprocal Rank Fusion (RRF)", () => {
  test("merges two lists and calculates correct RRF scores", () => {
    const list1: RankedResult[] = [
      { id: "doc1" }, // rank: 1
      { id: "doc2" }, // rank: 2
      { id: "doc3" }, // rank: 3
    ];

    const list2: RankedResult[] = [
      { id: "doc2" }, // rank: 1
      { id: "doc4" }, // rank: 2
      { id: "doc1" }, // rank: 3
    ];

    const k = 60;
    const results = rrf([list1, list2], k);

    expect(results.length).toBe(4);

    // doc2: (1 / (60+2)) + (1 / (60+1))
    const expectedDoc2Score = 1 / 62 + 1 / 61;
    // doc1: (1 / (60+1)) + (1 / (60+3))
    const expectedDoc1Score = 1 / 61 + 1 / 63;
    // doc3: 1 / 63
    const expectedDoc3Score = 1 / 63;
    // doc4: 1 / 62
    const expectedDoc4Score = 1 / 62;

    // Check sorting
    expect(results[0].id).toBe("doc2");
    expect(results[1].id).toBe("doc1");
    expect(results[2].id).toBe("doc4");
    expect(results[3].id).toBe("doc3");

    // Check scores
    expect(results[0].score).toBeCloseTo(expectedDoc2Score, 5);
    expect(results[1].score).toBeCloseTo(expectedDoc1Score, 5);
    expect(results[2].score).toBeCloseTo(expectedDoc4Score, 5);
    expect(results[3].score).toBeCloseTo(expectedDoc3Score, 5);
  });

  test("uses explicit rank property if provided", () => {
    const list: RankedResult[] = [
      { id: "docA", rank: 5 },
      { id: "docB", rank: 10 },
    ];

    const results = rrf([list], 0); // k=0 for simple math
    expect(results[0].id).toBe("docA");
    expect(results[0].score).toBeCloseTo(1 / 5, 5);
    expect(results[1].id).toBe("docB");
    expect(results[1].score).toBeCloseTo(1 / 10, 5);
  });

  test("merges metadata", () => {
    const list1: RankedResult[] = [
      { id: "doc1", metadata: { source: "dense", title: "Hello" } },
    ];
    const list2: RankedResult[] = [
      { id: "doc1", metadata: { source: "sparse", tag: "news" } },
    ];

    const results = rrf([list1, list2]);
    expect(results[0].metadata).toEqual({
      source: "sparse", // list2 overrides list1 for 'source'
      title: "Hello",
      tag: "news",
    });
  });
});

describe("Relative Score Fusion (RSF)", () => {
  test("min-max normalizes scores and applies weights", () => {
    // Range: 10 to 50
    const list1: RankedResult[] = [
      { id: "doc1", score: 50 }, // norm: 1.0
      { id: "doc2", score: 30 }, // norm: 0.5
      { id: "doc3", score: 10 }, // norm: 0.0
    ];

    // Range: 0.1 to 0.9
    const list2: RankedResult[] = [
      { id: "doc2", score: 0.9 }, // norm: 1.0
      { id: "doc4", score: 0.5 }, // norm: 0.5
      { id: "doc1", score: 0.1 }, // norm: 0.0
    ];

    // Weight: 0.7 for list1, 0.3 for list2
    const results = rsf([list1, list2], [0.7, 0.3]);

    expect(results.length).toBe(4);

    // doc1: (1.0 * 0.7) + (0.0 * 0.3) = 0.7
    // doc2: (0.5 * 0.7) + (1.0 * 0.3) = 0.35 + 0.3 = 0.65
    // doc4: (0.5 * 0.3) = 0.15
    // doc3: (0.0 * 0.7) = 0.0

    expect(results[0].id).toBe("doc1");
    expect(results[0].score).toBeCloseTo(0.7, 5);

    expect(results[1].id).toBe("doc2");
    expect(results[1].score).toBeCloseTo(0.65, 5);

    expect(results[2].id).toBe("doc4");
    expect(results[2].score).toBeCloseTo(0.15, 5);

    expect(results[3].id).toBe("doc3");
    expect(results[3].score).toBeCloseTo(0.0, 5);
  });

  test("handles single element lists properly", () => {
    const list1: RankedResult[] = [{ id: "doc1", score: 100 }];
    const results = rsf([list1]);

    // Range is 0, so normalized score should be 1.0
    expect(results[0].score).toBe(1.0);
  });

  test("throws error if weights length mismatches resultSets length", () => {
    const list: RankedResult[] = [{ id: "doc1", score: 10 }];
    expect(() => rsf([list], [0.5, 0.5])).toThrow();
  });
});
