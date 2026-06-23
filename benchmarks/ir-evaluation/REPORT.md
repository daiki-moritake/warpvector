# IR精度評価レポート

> 自動生成: 2026-06-22 | コーパス: 200ドキュメント (2ドメイン×50 + 60クロス + 40一般) | クエリ: 30件

## 概要

WarpVector の各アダプタが情報検索タスクの精度に与える影響を、
NDCG@K（Normalized Discounted Cumulative Gain）と MRR（Mean Reciprocal Rank）で定量評価しました。

## 結果

| 手法 | 次元 | NDCG@10 | NDCG@50 | MRR |
|------|------|---------|---------|-----|
| Vanilla (no transform) | 64 | 54.7% | 54.4% | 78.3% |
| Intent Warping (manual) | 64 | 58.6% | 56.5% | 78.7% |
| IntentMatrixFactory (auto) | 64 | 57.8% | 53.6% | 70.9% |
| Vanilla (no transform) | 256 | 68.2% | 62.8% | 81.4% |
| Intent Warping (manual) | 256 | 77.0% | 68.9% | 89.4% |
| IntentMatrixFactory (auto) | 256 | 68.7% | 58.3% | 92.8% |
| Vanilla (no transform) | 768 | 81.1% | 72.3% | 85.6% |
| Intent Warping (manual) | 768 | 88.0% | 78.7% | 91.7% |
| IntentMatrixFactory (auto) | 768 | 50.3% | 52.8% | 55.6% |

## Vanilla ベースライン対比の改善率

### 64次元

| 手法 | NDCG@10 改善 | MRR 改善 |
|------|-------------|---------|
| Intent Warping (manual) | +7.2% | +0.5% |
| IntentMatrixFactory (auto) | +5.7% | -9.5% |

### 256次元

| 手法 | NDCG@10 改善 | MRR 改善 |
|------|-------------|---------|
| Intent Warping (manual) | +13.0% | +9.9% |
| IntentMatrixFactory (auto) | +0.9% | +14.0% |

### 768次元

| 手法 | NDCG@10 改善 | MRR 改善 |
|------|-------------|---------|
| Intent Warping (manual) | +8.5% | +7.1% |
| IntentMatrixFactory (auto) | -38.0% | -35.1% |

## 推奨事項


---

*このレポートは `bun run benchmarks/ir-evaluation/evaluate.ts` で自動生成されました。*