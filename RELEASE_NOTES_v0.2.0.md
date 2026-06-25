# 🌊 WarpVector v0.2.0 — Wave-Inspired Rerankers, Interactive Playground & create-warpvector-app / 波動から着想を得たリランカー、インタラクティブプレイグラウンド

WarpVector v0.2.0 is a major feature release with **7 new ML adapters**, an **interactive browser playground**, and a **project scaffolding CLI** — all while maintaining zero dependencies and sub-millisecond inference.

WarpVector v0.2.0 はメジャー機能リリースです。ゼロ依存性とサブミリ秒の推論速度を維持したまま、**7個の新しいMLアダプター**、**インタラクティブなブラウザプレイグラウンド**、および**プロジェクト自動生成CLI**を追加しました。

---

## 🇬🇧 English Release Notes

### ✨ Highlights
- 🌊 **Wave-inspired rerankers** — `TimeReversalReranker` and `MultipathScatteringReranker` bring physics-inspired graph algorithms to vector search reranking.
- 🎮 **[Interactive Playground](https://daiki-moritake.github.io/warpvector/)** — Visualize vector space transformations in real-time with PCA projection, quantization controls, and live code snippets.
- 🚀 **`npx create-warpvector-app`** — Scaffold a new WarpVector project in seconds.
- 🇯🇵 **Full Japanese documentation** — API Reference, Tutorial, Cookbooks, and more.

### 🧠 New ML Adapters & Trainers

| Adapter / Trainer | Description |
|---|---|
| `SoftWhiteningAdapter` | Inverse-diffusion-based soft whitening for online dimensionality tuning. |
| `TimeReversalReranker` | Wave-equation-inspired graph-based reranking. |
| `MultipathScatteringReranker` | Random-walk hub detection reranker (WASM-accelerated). |
| `BaseGraphReranker` | Shared base class for graph-based rerankers. |
| `MoEAdapter` | Mixture of Experts routing adapter. |
| `AutoTuningPipeline` | AutoML-powered automatic pipeline optimization. |
| `CrossEncoderTrainer` | High-accuracy cross-encoder fine-tuning trainer. |

### 🛠️ Developer Experience
- **`create-warpvector-app` CLI** — `npx create-warpvector-app` to bootstrap projects instantly.
- **Interactive Playground** — PCA-based 2D projection camera, LLM-powered embedding generation, blend mode visualization, quantization/whitening controls, and cost simulation visuals.
- **New examples** — Express Batch API and Edge API Handler samples.
- **Enhanced CI** — Node.js 20/22 compatibility matrix, automated bundle size monitoring.

### 📚 Documentation
- 📖 [Migration Guide (v0.1 → v0.2)](https://github.com/daiki-moritake/warpvector/blob/main/docs/migration-guide.md) — No breaking changes! Existing code works as-is.
- 🍳 3 new Cookbooks: [E-commerce Search](https://github.com/daiki-moritake/warpvector/blob/main/docs/cookbook/ecommerce-search.md), [Pinecone RAG](https://github.com/daiki-moritake/warpvector/blob/main/docs/cookbook/rag-with-pinecone.md), [Cloudflare Edge](https://github.com/daiki-moritake/warpvector/blob/main/docs/cookbook/edge-cloudflare.md)
- 🇯🇵 Full Japanese translations for all documentation.

### ⚙️ Changed
- Renamed `InverseDiffusionAdapter` → `SoftWhiteningAdapter`.
- Updated CI Node.js version to v22.
- Optimized WASM binary storage with fixed-length byte arrays.
- Improved Trainer and Pipeline adapter interfaces for batch processing.
- Localized validation error messages.

### 📊 Stats
- **Tests**: 209 passed (147 → 209, +62)
- **Bundle size**: 112KB ESM / 24KB gzipped
- **Package size**: 194 kB (22 files)

---

## 🇯🇵 日本語リリースノート

### ✨ ハイライト
- 🌊 **波動から着想を得たリランカー** — 物理学のグラフアルゴリズムを取り入れた `TimeReversalReranker` (時間反転リランカー) および `MultipathScatteringReranker` (多重経路散乱リランカー) を追加しました。
- 🎮 **[インタラクティブプレイグラウンド](https://daiki-moritake.github.io/warpvector/)** — PCAプロジェクション、量子化コントロール、リアルタイムのコードスニペットを使用して、ベクトル空間の変換をブラウザ上で視覚化できます。
- 🚀 **`npx create-warpvector-app`** — わずか数秒で新しい WarpVector プロジェクトの雛形を作成可能に。
- 🇯🇵 **ドキュメントの完全日本語化** — API リファレンス、チュートリアル、クックブックなど、すべての公式ドキュメントを日本語化しました。

### 🧠 新しいMLアダプター＆トレーナー

| アダプター / トレーナー | 説明 |
|---|---|
| `SoftWhiteningAdapter` | オンラインでの次元調整を行う、逆拡散に基づくソフト白色化アダプター。 |
| `TimeReversalReranker` | 波動方程式から着想を得たグラフベースのリランカー。 |
| `MultipathScatteringReranker` | ランダムウォークによるハブ検出を利用したリランカー (WASM加速)。 |
| `BaseGraphReranker` | グラフベースリランカーの共通ベースクラス。 |
| `MoEAdapter` | 専門家混合（Mixture of Experts）ルーティングアダプター。 |
| `AutoTuningPipeline` | AutoMLを活用した自動パイプライン最適化機能。 |
| `CrossEncoderTrainer` | 高精度なクロスエンコーダーのファインチューニングを行うトレーナー。 |

### 🛠️ デベロッパーエクスペリエンス (DX)
- **`create-warpvector-app` CLI** — `npx create-warpvector-app` でプロジェクトを瞬時に立ち上げ。
- **インタラクティブプレイグラウンド** — PCAベースの2Dプロジェクションカメラ、LLMを使用した埋め込み生成、ブレンドモードの可視化、量子化/白色化コントロール、コストシミュレーション。
- **新しいサンプルコード** — Express Batch API と Edge API ハンドラーのサンプルを追加。
- **CIの強化** — Node.js 20/22 の互換性マトリクスと、バンドルサイズの自動監視。

### 📚 ドキュメント
- 📖 [移行ガイド (v0.1 → v0.2)](https://github.com/daiki-moritake/warpvector/blob/main/docs/migration-guide.md) — 破壊的変更はありません。既存のコードはそのまま動作します。
- 🍳 3つのクックブックを新規追加: [Eコマース検索](https://github.com/daiki-moritake/warpvector/blob/main/docs/cookbook/ecommerce-search.md), [PineconeによるRAG](https://github.com/daiki-moritake/warpvector/blob/main/docs/cookbook/rag-with-pinecone.md), [Cloudflare Edge](https://github.com/daiki-moritake/warpvector/blob/main/docs/cookbook/edge-cloudflare.md)

---

### 📦 インストール / アップグレード

```bash
npm install warpvector@0.2.0
```
