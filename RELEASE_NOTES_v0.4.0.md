# 🚀 WarpVector v0.4.0 — Architecture Refinement & Evaluation Kit / アーキテクチャ再設計 & 評価キット

WarpVector `v0.4.0` is a significant release that introduces three new packages, formalizes the separation between edge inference and backend-heavy tasks, and adds a built-in RAG evaluation toolkit.

WarpVector `v0.4.0` は、3つの新パッケージの追加、エッジ推論とバックエンド重処理の明確な分離、そしてRAG評価ツールキットの導入を含む重要なリリースです。

---

## 🇬🇧 English Release Notes

### ✨ Highlights

#### 🏗️ Architecture Refinement: Edge vs. Backend Separation

WarpVector now formally separates **edge-ready inference** modules from **backend-heavy** training and evaluation tasks:

| Layer | Packages | Purpose |
|-------|----------|---------|
| **Edge Inference** | `@warpvector/core`, `@warpvector/ml`, `@warpvector/extras` | Lightweight, zero-dependency, WASM-accelerated — runs on Cloudflare Workers, Vercel Edge, Deno Deploy |
| **Reranking** | `@warpvector/rerank` | ColBERT late interaction reranking — can run on edge or server |
| **Backend Training** | `@warpvector/train` | SoftWhitening, TripletTrainer, fine-tuning — designed for Node.js/Bun server environments |
| **Evaluation** | `@warpvector/eval` | RAG pipeline evaluation with Precision@K, NDCG@K, MRR, MAP — CLI + programmatic API |
| **Integrations** | `@warpvector/prisma`, `@warpvector/langchain` | Database and framework integrations |

#### 📦 New Package: `@warpvector/train`

Training and fine-tuning utilities have been extracted from `@warpvector/ml` into a dedicated package:

- `SoftWhiteningAdapter` — Anisotropy correction with learnable parameters
- `TripletTrainer` — Metric learning with triplet loss
- `BaseTrainer` — Foundation for custom training loops
- All training-related adapters and optimizers

```typescript
import { SoftWhiteningAdapter } from 'warpvector/train';
// or
import { SoftWhiteningAdapter } from '@warpvector/train';
```

#### 📦 New Package: `@warpvector/rerank`

ColBERT-based late interaction reranking, cleanly separated:

```typescript
import { ColbertAdapter } from 'warpvector/rerank';
// or
import { ColbertAdapter } from '@warpvector/rerank';
```

#### 📦 New Package: `@warpvector/eval`

A built-in RAG evaluation toolkit for measuring retrieval quality:

```typescript
import { WarpEvaluator } from '@warpvector/eval';

const evaluator = new WarpEvaluator({ k: 10 });
const results = evaluator.evaluate(predictions, groundTruth);
// → { precision: 0.85, recall: 0.92, ndcg: 0.89, mrr: 0.91, map: 0.87 }
```

CLI usage:
```bash
npx warpvector-eval --input results.json --k 10 --format markdown
```

#### 🔧 API Unification: QuantizationAdapter

The `QuantizationAdapter` API has been unified around the `encode()` method:

```diff
// Before (v0.3.x)
- const quantized = adapter.tune(vector);
+ const quantized = adapter.encode(vector);
```

#### 🎨 Warpvector Studio (Playground) Upgrade

- Visualization engine rewritten with animation state snapshots
- Improved canvas rendering performance
- Enhanced UI helper utilities

### ⚠️ Breaking Changes

| Change | Migration |
|--------|-----------|
| `QuantizationAdapter.tune()` removed | Use `encode()` instead |
| `SoftWhiteningAdapter` moved to `@warpvector/train` | Update import: `from 'warpvector/train'` |
| `TripletTrainer` moved to `@warpvector/train` | Update import: `from 'warpvector/train'` |
| `ColbertAdapter` moved to `@warpvector/rerank` | Update import: `from 'warpvector/rerank'` |
| Adam optimizer removed from `@warpvector/ml` | Use `@warpvector/train` instead |
| WarpPipeline final stage handling updated | Use `setFinalStage()` with `encode`-based adapters |

---

## 🇯🇵 日本語リリースノート

### ✨ ハイライト

#### 🏗️ アーキテクチャ再設計：エッジ推論とバックエンドの明確な分離

WarpVectorは、**エッジ対応の推論モジュール**と**バックエンド向けのトレーニング・評価タスク**を明確に分離しました：

| レイヤー | パッケージ | 用途 |
|---------|-----------|------|
| **エッジ推論** | `@warpvector/core`, `@warpvector/ml`, `@warpvector/extras` | 軽量・ゼロ依存・WASM高速化 — Cloudflare Workers、Vercel Edge、Deno Deployで動作 |
| **リランキング** | `@warpvector/rerank` | ColBERT後期相互作用リランキング — エッジ・サーバー両対応 |
| **バックエンドトレーニング** | `@warpvector/train` | SoftWhitening、TripletTrainer、ファインチューニング — Node.js/Bunサーバー環境向け |
| **評価** | `@warpvector/eval` | RAGパイプライン評価（Precision@K, NDCG@K, MRR, MAP）— CLI + プログラマティックAPI |
| **インテグレーション** | `@warpvector/prisma`, `@warpvector/langchain` | データベース・フレームワーク連携 |

#### 📦 新パッケージ: `@warpvector/train`

トレーニングおよびファインチューニング関連のユーティリティが `@warpvector/ml` から専用パッケージとして分離されました：

- `SoftWhiteningAdapter` — 学習可能パラメータによるAnisotropy補正
- `TripletTrainer` — Triplet Lossによるメトリック学習
- `BaseTrainer` — カスタムトレーニングループの基盤
- すべてのトレーニング関連アダプターとオプティマイザー

#### 📦 新パッケージ: `@warpvector/rerank`

ColBERTベースの後期相互作用リランキングを独立パッケージとして分離：

```typescript
import { ColbertAdapter } from 'warpvector/rerank';
```

#### 📦 新パッケージ: `@warpvector/eval`

RAGパイプラインの検索品質を測定する組み込み評価ツールキット：

```typescript
import { WarpEvaluator } from '@warpvector/eval';

const evaluator = new WarpEvaluator({ k: 10 });
const results = evaluator.evaluate(predictions, groundTruth);
// → { precision: 0.85, recall: 0.92, ndcg: 0.89, mrr: 0.91, map: 0.87 }
```

CLI使用例：
```bash
npx warpvector-eval --input results.json --k 10 --format markdown
```

#### 🔧 API統一: QuantizationAdapter

`QuantizationAdapter` のAPIが `encode()` メソッドに統一されました：

```diff
// 以前 (v0.3.x)
- const quantized = adapter.tune(vector);
// 現在 (v0.4.0)
+ const quantized = adapter.encode(vector);
```

#### 🎨 Warpvector Studio（Playground）アップグレード

- アニメーション状態スナップショットによるビジュアライゼーションエンジンの刷新
- キャンバスレンダリングパフォーマンスの改善
- UIヘルパーユーティリティの強化

### ⚠️ 破壊的変更

| 変更内容 | 移行方法 |
|---------|---------|
| `QuantizationAdapter.tune()` 削除 | `encode()` を使用 |
| `SoftWhiteningAdapter` を `@warpvector/train` へ移動 | インポート変更: `from 'warpvector/train'` |
| `TripletTrainer` を `@warpvector/train` へ移動 | インポート変更: `from 'warpvector/train'` |
| `ColbertAdapter` を `@warpvector/rerank` へ移動 | インポート変更: `from 'warpvector/rerank'` |
| Adam optimizer を `@warpvector/ml` から削除 | `@warpvector/train` を使用 |
| WarpPipeline final stage処理を更新 | `setFinalStage()` と `encode` ベースのアダプターを使用 |

---

### 📊 統計 / Statistics

| Metric | Value |
|--------|-------|
| Commits since v0.3.2 | 13 |
| Files changed | 123 |
| Lines added | +2,135 |
| Lines removed | -739 |
| Total packages | 10 |
| Tests | 297 passed ✅ |
| expect() calls | 1,579 |
| Test files | 44 |

---

### 🧪 テスト検証 / Tests

- **297 tests** / 44 files / **1,579 expect()** — All Passed ✅
- Architecture separation verified: edge modules remain zero-dependency
- Full build pipeline validated across all 10 packages

---

### ⬆️ アップグレード方法 / How to Upgrade

```bash
npm install warpvector@0.4.0
# または / or
bun add warpvector@0.4.0
```

**⚠️ Breaking changes exist.** See the migration table above for import path changes.
**⚠️ 破壊的変更があります。** 上記の移行テーブルを参照してインポートパスを更新してください。

---

### 📦 Full Package List

| Package | Version | Description |
|---------|---------|-------------|
| `warpvector` | 0.4.0 | Main bundle (all-in-one) |
| `@warpvector/core` | 0.4.0 | Core pipeline, adapters, WASM runtime |
| `@warpvector/ml` | 0.4.0 | ML adapters (Intent, LoRA, MLP, MoE) |
| `@warpvector/train` | 0.4.0 | Training & fine-tuning (SoftWhitening, Triplet) |
| `@warpvector/rerank` | 0.4.0 | ColBERT reranking |
| `@warpvector/eval` | 0.4.0 | RAG evaluation toolkit |
| `@warpvector/extras` | 0.4.0 | Quantization, VSA, Safety adapters |
| `@warpvector/experimental` | 0.4.0 | Experimental features |
| `@warpvector/prisma` | 0.4.0 | Prisma/pgvector integration |
| `@warpvector/langchain` | 0.4.0 | LangChain integration |
| `create-warpvector-app` | 0.4.0 | Project scaffolding CLI |
