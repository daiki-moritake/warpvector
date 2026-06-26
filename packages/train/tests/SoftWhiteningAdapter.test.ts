import { expect, test, describe, beforeAll } from "bun:test";
import { SoftWhiteningAdapter } from "../src/adapters/SoftWhiteningAdapter";
import { initWasm, innerProduct, normalize } from "@warpvector/core";

describe("SoftWhiteningAdapter", () => {
  beforeAll(async () => {
    await initWasm();
  });

  test("should initialize correctly", () => {
    const adapter = new SoftWhiteningAdapter(10, {
      learningRate: 0.05,
      numComponents: 3,
      tau: 2.0,
    });

    expect(adapter.dim).toBe(10);
    expect(adapter.components.length).toBe(3);
    expect(adapter.eigenvalues.length).toBe(3);
    expect(adapter.tau).toBe(2.0);
    expect(adapter.eigenvalues[0]).toBeCloseTo(1e-6);
  });

  test("should throw error for invalid parameters", () => {
    expect(() => new SoftWhiteningAdapter(10, { tau: -1.0 })).toThrow("non-negative");
    expect(() => new SoftWhiteningAdapter(10, { numComponents: 0 })).toThrow("positive");
  });

  test("should track eigenvalues and update components online", () => {
    const adapter = new SoftWhiteningAdapter(5, {
      learningRate: 0.1,
      numComponents: 2,
    });

    // Create a dataset with high variance along [1, 1, 1, 1, 1] but zero mean
    const bias = normalize(new Float32Array([1, 1, 1, 1, 1]));

    for (let i = 0; i < 100; i++) {
      const vec = new Float32Array(5);
      const sign = Math.random() > 0.5 ? 1 : -1;
      for (let j = 0; j < 5; j++) {
        vec[j] = bias[j] * sign * (0.8 + Math.random() * 0.4) + (Math.random() - 0.5) * 0.1;
      }
      adapter.update(vec);
    }

    // The first eigenvalue should have grown significantly (close to 1.0)
    expect(adapter.eigenvalues[0]).toBeGreaterThan(0.5);
    
    // The top component should be aligned with the bias
    const alignment = Math.abs(innerProduct(adapter.components[0], bias));
    expect(alignment).toBeGreaterThan(0.9);
  });

  test("should sharpen vector smoothly using inverse heat kernel (tau)", () => {
    const adapter = new SoftWhiteningAdapter(4, {
      learningRate: 0.1,
      numComponents: 1,
      tau: 5.0, // High tau for strong sharpening
    });

    // Force learning on a specific direction with zero mean
    const bias = new Float32Array([1, 0, 0, 0]);
    for (let i = 0; i < 100; i++) {
      const sign = Math.random() > 0.5 ? 1 : -1;
      adapter.update([sign * (1 + Math.random() * 0.1), (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1]);
    }

    // Now test sharpening
    // A vector that is mostly bias with a bit of orthogonal signal
    const testVec = new Float32Array([1.0, 0.5, 0, 0]);
    
    // Original projection on bias is large
    const origProj = testVec[0] - adapter.mean[0]; 
    const origOrtho = testVec[1] - adapter.mean[1];
    
    const sharpVec = adapter.tune(testVec);
    
    // The sharp vector should have reduced the dominant component significantly
    // and preserved the orthogonal component
    const sharpProj = innerProduct(sharpVec, adapter.components[0]);
    const sharpOrtho = sharpVec[1]; // Since component 0 is roughly [1,0,0,0], index 1 is orthogonal
    
    // Relative strength of orthogonal component should be higher now
    const origRatio = Math.abs(origOrtho / origProj);
    const sharpRatio = Math.abs(sharpOrtho / sharpProj);
    
    expect(sharpRatio).toBeGreaterThan(origRatio);
  });

  test("should serialize and deserialize state", () => {
    const adapter = new SoftWhiteningAdapter(3, {
      learningRate: 0.05,
      numComponents: 2,
      tau: 1.5,
    });

    adapter.update([1, 2, 3]);
    adapter.update([-1, 0, 1]);

    const stateStr = adapter.exportState();
    const restored = SoftWhiteningAdapter.importState(stateStr);

    expect(restored.dim).toBe(adapter.dim);
    expect(restored.tau).toBe(adapter.tau);
    expect(restored.normalizeOutput).toBe(adapter.normalizeOutput);
    expect(restored.mean).toEqual(adapter.mean);
    expect(restored.eigenvalues).toEqual(adapter.eigenvalues);
    expect(restored.components.length).toBe(adapter.components.length);
  });
  test("should normalize output when normalizeOutput is true", () => {
    const adapter = new SoftWhiteningAdapter(4, {
      tau: 1.0,
      numComponents: 1,
      normalizeOutput: true,
    });
    const vec = new Float32Array([1, 2, 3, 4]);
    const tuned = adapter.tune(vec);
    
    // Check if the norm is 1.0
    const norm = Math.sqrt(tuned.reduce((sum, v) => sum + v * v, 0));
    expect(norm).toBeCloseTo(1.0, 5);
  });

  test("tuneBatch should produce identical results to tune in loop", () => {
    const adapter = new SoftWhiteningAdapter(3, {
      tau: 2.0,
      numComponents: 2,
      normalizeOutput: true,
    });
    // Fill eigenvalues and components with some data
    adapter.eigenvalues[0] = 0.5;
    adapter.eigenvalues[1] = 0.2;
    adapter.components[0] = normalize(new Float32Array([1, -1, 0]));
    adapter.components[1] = normalize(new Float32Array([1, 1, 1]));

    const vectors = [
      new Float32Array([1, 0, 0]),
      new Float32Array([0, 1, 0]),
      new Float32Array([0, 0, 1]),
      new Float32Array([0.5, -0.5, 0.5]),
    ];

    const batchResults = adapter.tuneBatch(vectors);
    const loopResults = vectors.map(v => adapter.tune(v));

    expect(batchResults.length).toBe(vectors.length);
    for (let i = 0; i < vectors.length; i++) {
      for (let j = 0; j < 3; j++) {
        expect(batchResults[i][j]).toBeCloseTo(loopResults[i][j], 5);
      }
    }
  });
});
