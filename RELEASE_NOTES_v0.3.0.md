# 🌌 WarpVector v0.3.0 — Auto-ML, Observability, Edge Integration

ベクトル検索ミドルウェア WarpVector の v0.3.0 をリリースしました。Intent行列の自動学習、本番運用向けのトレーシング、Cloudflare Vectorize ネイティブ統合を追加し、プロダクション対応を大幅に強化しています。

---

## ✨ ハイライト

### 🧠 IntentMatrixFactory — Intent行列の自動生成 (Auto-ML)

手動で変換行列を設計する必要がなくなりました。カテゴリごとのサンプルベクトルを与えるだけで、InfoNCE対照学習により最適なIntent行列を自動生成します。

```typescript
import { IntentMatrixFactory } from 'warpvector/ml';

const factory = new IntentMatrixFactory(1536);
factory.addCategory("tech", [techVec1, techVec2, techVec3]);
factory.addCategory("business", [bizVec1, bizVec2, bizVec3]);

const intents = await factory.build(); // InfoNCE で自動学習
```

### 📊 WarpTracer — OpenTelemetry 互換トレーシング

ゼロ依存の軽量トレーサーを内蔵。本番環境でのレイテンシ計測やボトルネック特定が可能です。

```typescript
import { WarpTracer } from 'warpvector';

const tracer = new WarpTracer();
const result = tracer.trace("intent.tune", { dim: 768 }, () => adapter.tune(vector, "tech"));
console.log(tracer.getMetrics());
```

### ☁️ Cloudflare Vectorize ネイティブ統合

Cloudflare Workers + Vectorize の組み合わせで、エッジでの超低遅延ベクトル検索が可能になりました。

```typescript
import { VectorDBAdapter } from 'warpvector';

const query = VectorDBAdapter.toVectorizeQuery(warpedVector, 10);
const results = await env.VECTORIZE_INDEX.query(query.vector, query.options);
```

---

## 🆕 新機能一覧

### Core (`@warpvector/core`)
- **WarpTracer**: `trace()` / `traceAsync()` / `getMetrics()` / `resetMetrics()`
- **VectorDBAdapter**: `toVectorizeQuery()` / `toVectorizeRecord()`
- **WarpPipeline**: `inspect()` / `dryRun()` / `metrics` API
- Graceful Degradation (次元不一致、未登録Intent等のエラーハンドリング)

### ML (`@warpvector/ml`)
- **IntentMatrixFactory**: `addCategory()` → `build()` で Intent 行列を自動学習

### CLI (`create-warpvector-app`)
- 🆕 **Minimal Intent Search** テンプレート
- 🆕 **Cloudflare Workers** テンプレート
- パッケージマネージャー選択 (npm / yarn / pnpm / bun)

### Experimental (`@warpvector/experimental`) — 新パッケージ
- ColBERT Late Interaction (MaxSim)
- Vector Symbolic Architecture (VSA)
- Anomaly Detection / Task Arithmetic

---

## 📈 パフォーマンス

| 指標 | 値 |
|------|------|
| MLP 推論 (WASM) | 1.1–3.8 µs/vector |
| Int8 量子化速度 | 322K vecs/sec |
| Binary 量子化速度 | 1.18M vecs/sec |
| パイプライン遅延 | 119 µs (Intent + Projection) |
| IR精度 (NDCG@10) | +13.0% 改善 (vs vanilla) |
| Int8 Recall@10 | 86–96% |
| メモリ削減 (Binary) | **96.9% 削減** |

---

## 🧪 テスト

- **290 テスト** / 43 ファイル / **1,562 expect()** — 全パス ✅
- 新規テストファイル:
  - `graceful-degradation.test.ts` (13 tests)
  - `telemetry-vectorize.test.ts` (12 tests)
  - `experimental.test.ts` (6 tests)
- 強化: `property-based.test.ts` (+15), `wasm-js-equivalence.test.ts` (+11)

---

## 📦 パッケージ

| パッケージ | バージョン |
|-----------|-----------|
| `warpvector` | 0.3.0 |
| `@warpvector/core` | 0.3.0 |
| `@warpvector/ml` | 0.3.0 |
| `@warpvector/extras` | 0.3.0 |
| `@warpvector/experimental` | 0.3.0 |
| `@warpvector/prisma` | 0.3.0 |
| `@warpvector/langchain` | 0.3.0 |
| `create-warpvector-app` | 0.3.0 |

---

## 📝 ドキュメント

- README 英語/日本語版を完全同期 (ベンチマーク結果、Feature Overview、デバッグ、Vectorize、WarpTracer を日本語版に追加)
- Prisma 統合サンプルを `sql-template-tag` 対応に修正 (README, docs 全4ファイル)
- Getting Started ガイド全面改訂 (EN/JA)
- IntentMatrixFactory 専用ドキュメント追加 (EN/JA)

---

## ⬆️ アップグレード方法

```bash
npm install warpvector@0.3.0
```

**破壊的変更はありません。** v0.2.0 からのドロップインアップグレードが可能です。

---

**Full Changelog**: `v0.2.0...v0.3.0`
