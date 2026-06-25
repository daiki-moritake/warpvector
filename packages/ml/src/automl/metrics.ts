import { InputVector, OutputVector, cosineSimilarity } from "@warpvector/core";

export type VectorType = InputVector | OutputVector;

export interface SearchExample<T = VectorType> {
  query: T;
  positive: T;
  negatives?: T[];
}

/**
 * コサイン類似度の計算
 */
export function computeCosineSimilarity(a: VectorType, b: VectorType): number {
  const vecA = (a instanceof Float32Array || Array.isArray(a)) ? a : new Float32Array(a as any);
  const vecB = (b instanceof Float32Array || Array.isArray(b)) ? b : new Float32Array(b as any);
  return cosineSimilarity(vecA, vecB);
}

/**
 * 1つのクエリに対する検索シミュレーションを行い、正解（Positive）の順位（1-indexed）を返します。
 * @param query クエリベクトル
 * @param positive 正解ベクトル
 * @param negatives 不正解ベクトルの配列
 * @returns 正解の順位（1が最上位）
 */
export function getPositiveRank(
  query: VectorType,
  positive: VectorType,
  negatives: VectorType[] = []
): number {
  const posScore = computeCosineSimilarity(query, positive);
  
  // 順位（自分よりスコアが高い negative の数 + 1）
  let rank = 1;
  for (const neg of negatives) {
    const negScore = computeCosineSimilarity(query, neg);
    if (negScore > posScore) {
      rank++;
    }
  }
  return rank;
}

/**
 * 検証用データセット全体の MRR (Mean Reciprocal Rank) を計算します。
 */
export function calculateMRR(dataset: SearchExample[]): number {
  if (dataset.length === 0) return 0;
  
  let mrrSum = 0;
  for (const example of dataset) {
    const rank = getPositiveRank(example.query, example.positive, example.negatives);
    mrrSum += 1.0 / rank;
  }
  
  return mrrSum / dataset.length;
}

/**
 * 検証用データセット全体の Recall@K を計算します。
 */
export function calculateRecall(dataset: SearchExample[], k: number): number {
  if (dataset.length === 0) return 0;
  
  let hits = 0;
  for (const example of dataset) {
    const rank = getPositiveRank(example.query, example.positive, example.negatives);
    if (rank <= k) {
      hits++;
    }
  }
  
  return hits / dataset.length;
}
