import { RankedResult, FusionResult } from "./rrf";

/**
 * Relative Score Fusion (RSF) を計算し、複数の検索結果リストを統合します。
 * 各リストのスコアをMin-Max正規化によって [0, 1] にスケーリングし、重み付け加算を行います。
 *
 * @param resultSets 検索結果リストの配列（例: [denseResults, sparseResults]）
 * @param weights 各検索結果に対する重み（例: [0.7, 0.3]）。省略時は全て1.0
 * @returns 重み付き正規化スコアの降順でソートされた統合結果の配列
 */
export function rsf(
  resultSets: RankedResult[][],
  weights?: number[]
): FusionResult[] {
  if (weights && weights.length !== resultSets.length) {
    throw new Error("Weights array length must match resultSets length");
  }

  const scoreMap = new Map<string | number, FusionResult>();

  for (let sIdx = 0; sIdx < resultSets.length; sIdx++) {
    const resultSet = resultSets[sIdx];
    const weight = weights ? weights[sIdx] : 1.0;

    if (resultSet.length === 0) continue;

    // 最小値・最大値の探索
    let min = Infinity;
    let max = -Infinity;
    for (const item of resultSet) {
      const score = item.score !== undefined ? item.score : 0;
      if (score < min) min = score;
      if (score > max) max = score;
    }

    const range = max - min;

    for (const item of resultSet) {
      const score = item.score !== undefined ? item.score : 0;
      // 要素が1つだけ、または全要素のスコアが等しい場合は最大スコア(1.0)とする
      const normalizedScore = range === 0 ? 1.0 : (score - min) / range;
      const weightedScore = normalizedScore * weight;

      if (scoreMap.has(item.id)) {
        const existing = scoreMap.get(item.id)!;
        existing.score += weightedScore;
        existing.metadata = { ...existing.metadata, ...item.metadata };
      } else {
        scoreMap.set(item.id, {
          id: item.id,
          score: weightedScore,
          metadata: item.metadata,
        });
      }
    }
  }

  // スコアの降順でソート
  return Array.from(scoreMap.values()).sort((a, b) => b.score - a.score);
}
