import { cosineSimilarity } from "@warpvector/core";

export interface CorpusItem {
  id: string;
  vector: number[] | Float32Array | Int8Array | Uint8Array;
}

export interface EvalQuery {
  queryVector: number[] | Float32Array | Int8Array | Uint8Array;
  expectedDocIds: string[];
}

export interface EvalConfig {
  corpus: CorpusItem[];
  dataset: EvalQuery[];
  kList: number[];
  pipeline?: {
    run: (vector: any, options?: any) => any;
  } | ((vector: any) => any);
  intentName?: string;
}

export interface MetricSummary {
  recall: Record<number, number>;
  ndcg: Record<number, number>;
  mrr: number;
  avgLatencyMs: number;
}

export interface EvalReport {
  vanilla: MetricSummary;
  warped: MetricSummary;
}

/**
 * 任意の型（Float32Array, Int8Array, Array等）を cosineSimilarity で使えるように
 * number[] または Float32Array にキャストします。
 */
function toFloatVector(vec: any): number[] | Float32Array {
  if (vec instanceof Float32Array || Array.isArray(vec)) {
    return vec;
  }
  return Array.from(vec);
}

/**
 * Recall@K を計算します。
 */
export function calculateRecall(retrievedIds: string[], expectedIds: string[], k: number): number {
  if (expectedIds.length === 0) return 0;
  const topK = retrievedIds.slice(0, k);
  const expectedSet = new Set(expectedIds);
  let hits = 0;
  for (const id of topK) {
    if (expectedSet.has(id)) {
      hits++;
    }
  }
  return hits / expectedIds.length;
}

/**
 * MRR (Mean Reciprocal Rank) を計算します。
 */
export function calculateMRR(retrievedIds: string[], expectedIds: string[]): number {
  if (expectedIds.length === 0) return 0;
  const expectedSet = new Set(expectedIds);
  for (let i = 0; i < retrievedIds.length; i++) {
    if (expectedSet.has(retrievedIds[i])) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * NDCG@K を計算します。
 */
export function calculateNDCG(retrievedIds: string[], expectedIds: string[], k: number): number {
  if (expectedIds.length === 0) return 0;
  const topK = retrievedIds.slice(0, k);
  const expectedSet = new Set(expectedIds);
  
  let dcg = 0;
  for (let i = 0; i < topK.length; i++) {
    if (expectedSet.has(topK[i])) {
      dcg += 1 / Math.log2(i + 2);
    }
  }

  let idcg = 0;
  const numRelevant = Math.min(k, expectedIds.length);
  for (let i = 0; i < numRelevant; i++) {
    idcg += 1 / Math.log2(i + 2);
  }

  if (idcg === 0) return 0;
  return dcg / idcg;
}

/**
 * コサイン類似度を用いてインメモリで簡易的なベクトル検索を実行します。
 */
export function searchInMemory(
  queryVec: number[] | Float32Array,
  corpus: { id: string; vector: number[] | Float32Array }[]
): { id: string; score: number }[] {
  const results = corpus.map(item => {
    const score = cosineSimilarity(queryVec, item.vector);
    return { id: item.id, score };
  });
  // 類似度の高い順にソート
  return results.sort((a, b) => b.score - a.score);
}

/**
 * Warpvector を適用した場合と適用しない場合の RAG 検索精度を評価・比較します。
 */
export async function evaluatePipeline(config: EvalConfig): Promise<EvalReport> {
  const kList = config.kList.length > 0 ? config.kList : [1, 3, 5, 10];
  const maxK = Math.max(...kList);

  // 1. キャスト済みのバニラ（オリジナル）コーパスの作成
  const vanillaCorpus = config.corpus.map(item => ({
    id: item.id,
    vector: toFloatVector(item.vector)
  }));

  // 2. パイプラインを適用したワープ済みのコーパスの作成
  let warpedCorpus = vanillaCorpus;
  const pipeline = config.pipeline;
  if (pipeline) {
    warpedCorpus = config.corpus.map(item => {
      let warpedVec;
      if (typeof pipeline === "function") {
        warpedVec = pipeline(item.vector);
      } else {
        warpedVec = pipeline.run(item.vector, { intent: config.intentName });
      }
      return {
        id: item.id,
        vector: toFloatVector(warpedVec)
      };
    });
  }

  // 3. 評価指標の初期化
  const vanillaMetrics = {
    recall: {} as Record<number, number>,
    ndcg: {} as Record<number, number>,
    mrr: 0,
    avgLatencyMs: 0
  };

  const warpedMetrics = {
    recall: {} as Record<number, number>,
    ndcg: {} as Record<number, number>,
    mrr: 0,
    avgLatencyMs: 0
  };

  for (const k of kList) {
    vanillaMetrics.recall[k] = 0;
    vanillaMetrics.ndcg[k] = 0;
    warpedMetrics.recall[k] = 0;
    warpedMetrics.ndcg[k] = 0;
  }

  let totalVanillaLatency = 0;
  let totalWarpedLatency = 0;

  // 4. データセット内の各クエリに対して評価を実行
  for (const item of config.dataset) {
    const vanillaQuery = toFloatVector(item.queryVector);

    // --- Vanilla 評価 ---
    const t0 = performance.now();
    const vanillaResults = searchInMemory(vanillaQuery, vanillaCorpus);
    const t1 = performance.now();
    totalVanillaLatency += (t1 - t0);

    const vanillaIds = vanillaResults.map(r => r.id);
    vanillaMetrics.mrr += calculateMRR(vanillaIds, item.expectedDocIds);
    for (const k of kList) {
      vanillaMetrics.recall[k] += calculateRecall(vanillaIds, item.expectedDocIds, k);
      vanillaMetrics.ndcg[k] += calculateNDCG(vanillaIds, item.expectedDocIds, k);
    }

    // --- Warped 評価 ---
    let warpedQuery = vanillaQuery;
    let tWarp0 = performance.now();
    if (pipeline) {
      if (typeof pipeline === "function") {
        warpedQuery = toFloatVector(pipeline(vanillaQuery));
      } else {
        warpedQuery = toFloatVector(pipeline.run(vanillaQuery, { intent: config.intentName }));
      }
    }
    const warpedResults = searchInMemory(warpedQuery, warpedCorpus);
    const tWarp1 = performance.now();
    totalWarpedLatency += (tWarp1 - tWarp0);

    const warpedIds = warpedResults.map(r => r.id);
    warpedMetrics.mrr += calculateMRR(warpedIds, item.expectedDocIds);
    for (const k of kList) {
      warpedMetrics.recall[k] += calculateRecall(warpedIds, item.expectedDocIds, k);
      warpedMetrics.ndcg[k] += calculateNDCG(warpedIds, item.expectedDocIds, k);
    }
  }

  const n = config.dataset.length;
  if (n > 0) {
    vanillaMetrics.mrr /= n;
    vanillaMetrics.avgLatencyMs = totalVanillaLatency / n;
    warpedMetrics.mrr /= n;
    warpedMetrics.avgLatencyMs = totalWarpedLatency / n;

    for (const k of kList) {
      vanillaMetrics.recall[k] /= n;
      vanillaMetrics.ndcg[k] /= n;
      warpedMetrics.recall[k] /= n;
      warpedMetrics.ndcg[k] /= n;
    }
  }

  return {
    vanilla: vanillaMetrics,
    warped: warpedMetrics
  };
}
