export interface RankedResult {
  id: string | number;
  score?: number;
  rank?: number;
  metadata?: Record<string, any>;
}

export interface FusionResult {
  id: string | number;
  score: number;
  metadata?: Record<string, any>;
}

/**
 * Reciprocal Rank Fusion (RRF) を計算し、複数の検索結果リストを統合します。
 * スコアの絶対値に依存せず、各リスト内での「順位」を用いて新しいスコアを算出します。
 *
 * @param resultSets 検索結果リストの配列（例: [denseResults, sparseResults]）
 * @param k RRFの平滑化定数（通常は60が使用されます）
 * @returns RRFスコアの降順でソートされた統合結果の配列
 */
export function rrf(
  resultSets: RankedResult[][],
  k: number = 60
): FusionResult[] {
  const scoreMap = new Map<string | number, FusionResult>();

  for (const resultSet of resultSets) {
    // 各resultSetは既にスコア等で降順ソートされていることを前提とします。
    for (let i = 0; i < resultSet.length; i++) {
      const item = resultSet[i];
      // rankプロパティがあればそれを、なければ配列のインデックス+1を順位とします。
      const rank = item.rank !== undefined ? item.rank : i + 1;
      const rrfScore = 1 / (k + rank);

      if (scoreMap.has(item.id)) {
        const existing = scoreMap.get(item.id)!;
        existing.score += rrfScore;
        // 既存のメタデータとマージ
        existing.metadata = { ...existing.metadata, ...item.metadata };
      } else {
        scoreMap.set(item.id, {
          id: item.id,
          score: rrfScore,
          metadata: item.metadata,
        });
      }
    }
  }

  // スコアの降順でソート
  return Array.from(scoreMap.values()).sort((a, b) => b.score - a.score);
}
