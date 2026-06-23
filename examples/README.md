# WarpVector Examples

このディレクトリには、`warpvector` の実用的なユースケースを示すサンプルコードが含まれています。
ローカルで簡単に実行して、動的なアフィン変換の威力を体験することができます。

## 実行方法

Bun または ts-node を使用して直接 TypeScript ファイルを実行できます。

```bash
# サンプル 1: ECサイトでの動的コンテキストルーティング
bun run examples/1-ecommerce-routing.ts

# サンプル 2: ユーザーフィードバックに基づくオンライン学習
bun run examples/2-online-learning.ts

# サンプル 3: モデルの次元削減 (Migration)
bun run examples/3-dimension-migration.ts
```

## サンプルの概要

### `1-ecommerce-routing.ts`
最も基本的な `IntentAdapter` の使い方です。同じ検索クエリでも、「学生」や「ゲーマー」といったコンテキスト（意図）の違いによって、ベクトル空間をどのように歪ませて検索結果を変えるかを示しています。

### `2-online-learning.ts`
`IntentTrainer` を使用した動的学習（SGD + Momentum）の例です。ユーザーが特定の商品をクリックした履歴（フィードバック）をもとに、検索ベクトルがその商品に近づくようにリアルタイムに行列とバイアスを学習・最適化します。

### `3-dimension-migration.ts`
`ProjectionAdapter` を使用した次元削減の例です。高次元モデルから出力されたベクトルを、非対称な行列を用いて軽量な次元に投影（プロジェクション）し、DBのマイグレーションコストを削減するユースケースを示しています。

### `auto-intent.ts` 🆕
`IntentMatrixFactory` を使った**Intent行列の自動生成**の完全な例です。カテゴリごとにサンプルベクトルを追加するだけで、InfoNCE対照学習により最適なアフィン変換行列を自動学習します。Vanilla検索 vs Intent Warping vs Auto-blending の検索結果の違いを体験できます。

```bash
bun run examples/auto-intent.ts
```
