/**
 * WarpVector Playground — Real Library Demo Engine
 *
 * This module uses the ACTUAL @warpvector/core library to perform
 * intent-based vector transformations. No simulation — real WASM-accelerated
 * affine transforms are happening in the browser.
 */
import { IntentAdapter, initWasm, type IntentWeights } from '@warpvector/core';

const DIM = 32; // Small enough for browser visualization, large enough for real math

/** Seed-based pseudo-random for reproducibility */
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

/** Generate a random unit vector */
function randomUnitVector(rng: () => number, dim: number): Float32Array {
  const v = new Float32Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    v[i] = rng() * 2 - 1;
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) v[i] /= norm;
  return v;
}

/** Generate a transformation matrix that pulls vectors in a category's direction */
function generateIntentMatrix(
  rng: () => number,
  dim: number,
  categoryDirection: Float32Array,
  strength: number,
): { matrix: Float32Array; bias: Float32Array } {
  // Start with identity + directional boost
  const matrix = new Float32Array(dim * dim);
  const bias = new Float32Array(dim);

  // Identity matrix
  for (let i = 0; i < dim; i++) matrix[i * dim + i] = 1.0;

  // Add outer product: strength * direction ⊗ direction
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      matrix[i * dim + j] += strength * categoryDirection[i] * categoryDirection[j];
    }
    bias[i] = strength * 0.1 * categoryDirection[i];
  }

  // Add small random perturbation for realism
  for (let i = 0; i < dim * dim; i++) {
    matrix[i] += (rng() - 0.5) * 0.02;
  }

  return { matrix, bias };
}

/** Cosine similarity between two vectors */
function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Project high-dimensional vector to 2D using two basis vectors */
function projectTo2D(
  vec: Float32Array,
  basis1: Float32Array,
  basis2: Float32Array,
): { x: number; y: number } {
  let x = 0, y = 0;
  for (let i = 0; i < vec.length; i++) {
    x += vec[i] * basis1[i];
    y += vec[i] * basis2[i];
  }
  return { x, y };
}

export interface DocPoint {
  id: number;
  name: string;
  category: string;
  color: string;
  baseVector: Float32Array;
  currentVector: Float32Array;
  pos: { x: number; y: number };
}

export interface DemoState {
  docs: DocPoint[];
  query: { vector: Float32Array; pos: { x: number; y: number } };
  adapter: IntentAdapter;
  basis1: Float32Array;
  basis2: Float32Array;
  wasmReady: boolean;
}

/** Create the full demo state with real warpvector IntentAdapter */
export async function createDemoState(lang: 'en' | 'ja'): Promise<DemoState> {
  // Initialize WASM
  const wasmResult = await initWasm();
  const wasmReady = wasmResult !== null;

  const rng = seededRandom(42);

  // Category direction vectors (define semantic regions in the vector space)
  const techDir = randomUnitVector(seededRandom(100), DIM);
  const bizDir = randomUnitVector(seededRandom(200), DIM);
  const medDir = randomUnitVector(seededRandom(300), DIM);
  const genDir = randomUnitVector(seededRandom(400), DIM);

  // Create the REAL IntentAdapter with REAL transformation matrices
  const techWeights = generateIntentMatrix(seededRandom(1001), DIM, techDir, 0.8);
  const bizWeights = generateIntentMatrix(seededRandom(1002), DIM, bizDir, 0.8);
  const medWeights = generateIntentMatrix(seededRandom(1003), DIM, medDir, 0.8);

  const adapter = new IntentAdapter(DIM);
  adapter.addIntent('technology', { matrix: techWeights.matrix, bias: techWeights.bias });
  adapter.addIntent('business', { matrix: bizWeights.matrix, bias: bizWeights.bias });
  adapter.addIntent('medical', { matrix: medWeights.matrix, bias: medWeights.bias });

  // Document names
  const docNames = lang === 'ja' ? {
    tech: ["TypeScript WASM ガイド", "React パフォーマンス最適化", "エッジコンピューティング設計", "ベクトルDB の内部構造"],
    business: ["Q3 収益分析レポート", "市場参入戦略", "エンタープライズ SaaS 価格設計", "スタートアップ資金調達ガイド"],
    medical: ["臨床試験デザイン", "薬物相互作用データベース", "患者データプライバシー", "ゲノミクスパイプライン"],
    general: ["機械学習入門", "データプライバシー規制", "クラウドインフラ構築"],
  } : {
    tech: ["TypeScript WASM Guide", "React Performance Tips", "Edge Computing Patterns", "Vector Database Internals"],
    business: ["Q3 Revenue Analysis", "Market Entry Strategy", "Enterprise SaaS Pricing", "Startup Fundraising Guide"],
    medical: ["Clinical Trial Design", "Drug Interaction Database", "Patient Data Privacy", "Genomics Pipeline"],
    general: ["Machine Learning Basics", "Data Privacy Compliance", "Cloud Infrastructure"],
  };

  // Generate document vectors with category-specific directions
  function makeDocVector(categoryDir: Float32Array, seed: number): Float32Array {
    const r = seededRandom(seed);
    const v = new Float32Array(DIM);
    let norm = 0;
    for (let i = 0; i < DIM; i++) {
      v[i] = categoryDir[i] * 0.7 + (r() - 0.5) * 0.6;
      norm += v[i] * v[i];
    }
    norm = Math.sqrt(norm);
    for (let i = 0; i < DIM; i++) v[i] /= norm;
    return v;
  }

  const docs: DocPoint[] = [];
  let id = 0;

  // Tech docs
  for (let i = 0; i < 4; i++) {
    docs.push({
      id: id++, name: docNames.tech[i], category: 'tech', color: '#3b82f6',
      baseVector: makeDocVector(techDir, 500 + i),
      currentVector: new Float32Array(DIM),
      pos: { x: 0, y: 0 },
    });
  }
  // Business docs
  for (let i = 0; i < 4; i++) {
    docs.push({
      id: id++, name: docNames.business[i], category: 'business', color: '#10b981',
      baseVector: makeDocVector(bizDir, 600 + i),
      currentVector: new Float32Array(DIM),
      pos: { x: 0, y: 0 },
    });
  }
  // Medical docs
  for (let i = 0; i < 4; i++) {
    docs.push({
      id: id++, name: docNames.medical[i], category: 'medical', color: '#f43f5e',
      baseVector: makeDocVector(medDir, 700 + i),
      currentVector: new Float32Array(DIM),
      pos: { x: 0, y: 0 },
    });
  }
  // General docs
  for (let i = 0; i < 3; i++) {
    docs.push({
      id: id++, name: docNames.general[i], category: 'general', color: '#94a3b8',
      baseVector: makeDocVector(genDir, 800 + i),
      currentVector: new Float32Array(DIM),
      pos: { x: 0, y: 0 },
    });
  }

  // Query vector (mixture of tech and general — ambiguous)
  const queryVec = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) {
    queryVec[i] = techDir[i] * 0.3 + bizDir[i] * 0.3 + genDir[i] * 0.3 + (rng() - 0.5) * 0.2;
  }
  let qNorm = 0;
  for (let i = 0; i < DIM; i++) qNorm += queryVec[i] * queryVec[i];
  qNorm = Math.sqrt(qNorm);
  for (let i = 0; i < DIM; i++) queryVec[i] /= qNorm;

  // 2D projection basis (PCA-like: use first two category directions)
  const basis1 = new Float32Array(DIM);
  const basis2 = new Float32Array(DIM);
  // Gram-Schmidt orthogonalization
  for (let i = 0; i < DIM; i++) basis1[i] = techDir[i] - bizDir[i];
  let b1Norm = 0;
  for (let i = 0; i < DIM; i++) b1Norm += basis1[i] * basis1[i];
  b1Norm = Math.sqrt(b1Norm);
  for (let i = 0; i < DIM; i++) basis1[i] /= b1Norm;

  let dot12 = 0;
  for (let i = 0; i < DIM; i++) {
    basis2[i] = medDir[i] - genDir[i];
    dot12 += basis2[i] * basis1[i];
  }
  for (let i = 0; i < DIM; i++) basis2[i] -= dot12 * basis1[i];
  let b2Norm = 0;
  for (let i = 0; i < DIM; i++) b2Norm += basis2[i] * basis2[i];
  b2Norm = Math.sqrt(b2Norm);
  for (let i = 0; i < DIM; i++) basis2[i] /= b2Norm;

  // Set initial positions (no intent = base vectors)
  for (const doc of docs) {
    doc.currentVector.set(doc.baseVector);
    doc.pos = projectTo2D(doc.baseVector, basis1, basis2);
  }

  const queryPos = projectTo2D(queryVec, basis1, basis2);

  return {
    docs,
    query: { vector: queryVec, pos: queryPos },
    adapter,
    basis1,
    basis2,
    wasmReady,
  };
}

/** Transform all document vectors using the REAL IntentAdapter */
export function transformWithIntent(
  state: DemoState,
  intent: string | null,
): { latencyMs: number; rankings: RankedDoc[] } {
  const t0 = performance.now();

  for (const doc of state.docs) {
    if (intent && intent !== 'none') {
      // REAL warpvector IntentAdapter.tune() — uses WASM if available
      doc.currentVector = state.adapter.tune(doc.baseVector, intent);
    } else {
      doc.currentVector = new Float32Array(doc.baseVector);
    }
    doc.pos = projectTo2D(doc.currentVector, state.basis1, state.basis2);
  }

  const latencyMs = performance.now() - t0;

  // Compute rankings by cosine similarity
  const rankings = state.docs.map((doc) => ({
    ...doc,
    score: cosineSim(doc.currentVector, state.query.vector),
  })).sort((a, b) => b.score - a.score);

  return { latencyMs, rankings };
}

/** Transform using BLENDED intents — demonstrates tuneBlended() */
export function transformWithBlend(
  state: DemoState,
  weights: Record<string, number>,
): { latencyMs: number; rankings: RankedDoc[]; codeSnippet: string } {
  // Filter out zero-weight intents
  const activeWeights: Record<string, number> = {};
  for (const [k, v] of Object.entries(weights)) {
    if (v > 0.01) activeWeights[k] = v;
  }

  const hasActive = Object.keys(activeWeights).length > 0;
  const t0 = performance.now();

  for (const doc of state.docs) {
    if (hasActive) {
      // REAL warpvector IntentAdapter.tuneBlended()
      doc.currentVector = state.adapter.tuneBlended(doc.baseVector, activeWeights);
    } else {
      doc.currentVector = new Float32Array(doc.baseVector);
    }
    doc.pos = projectTo2D(doc.currentVector, state.basis1, state.basis2);
  }

  const latencyMs = performance.now() - t0;

  const rankings = state.docs.map((doc) => ({
    ...doc,
    score: cosineSim(doc.currentVector, state.query.vector),
  })).sort((a, b) => b.score - a.score);

  // Generate the actual code snippet being executed
  const weightsStr = Object.entries(activeWeights)
    .map(([k, v]) => `  ${k}: ${v.toFixed(2)}`)
    .join(',\n');
  const codeSnippet = hasActive
    ? `const warped = adapter.tuneBlended(\n  baseVector,\n  {\n${weightsStr}\n  }\n);`
    : `// No intent applied\nconst result = baseVector;`;

  return { latencyMs, rankings, codeSnippet };
}

/** Run batch benchmark — demonstrates tuneBatch() performance */
export function runBenchmark(
  state: DemoState,
  batchSize: number = 1000,
): BenchmarkResult {
  const dim = 32;
  const rng = seededRandom(9999);

  // Generate batch vectors
  const vectors: Float32Array[] = [];
  for (let i = 0; i < batchSize; i++) {
    vectors.push(randomUnitVector(rng, dim));
  }

  // Benchmark tuneBatch (WASM path)
  const t0 = performance.now();
  state.adapter.tuneBatch(vectors, 'technology');
  const batchMs = performance.now() - t0;

  // Benchmark individual tune calls
  const t1 = performance.now();
  for (let i = 0; i < batchSize; i++) {
    state.adapter.tune(vectors[i], 'technology');
  }
  const individualMs = performance.now() - t1;

  return {
    batchSize,
    batchMs,
    individualMs,
    batchOpsPerSec: Math.round(batchSize / (batchMs / 1000)),
    individualOpsPerSec: Math.round(batchSize / (individualMs / 1000)),
    speedup: individualMs / batchMs,
  };
}

export interface BenchmarkResult {
  batchSize: number;
  batchMs: number;
  individualMs: number;
  batchOpsPerSec: number;
  individualOpsPerSec: number;
  speedup: number;
}

export interface RankedDoc extends DocPoint {
  score: number;
}

export { cosineSim, projectTo2D, DIM };
