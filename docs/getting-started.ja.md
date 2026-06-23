# Getting Started (はじめに)

`warpvector` は、ベクトル検索における「ユーザーの意図」に合わせたベクトル空間の動的変形（アフィン変換）を可能にする超軽量ミドルウェアです。

## 1. インストール

パッケージマネージャーを使用してインストールします。

```bash
# npmを使用する場合
npm install warpvector

# bunを使用する場合
bun add warpvector
```

Python環境や重厚な機械学習ライブラリ（PyTorchなど）は一切不要です。内部で自動的に WebAssembly (WASM) がロードされ、大規模なベクトルバッチ処理も高速に処理されます。

---

## 2. 最短ルート: Intent行列の自動生成

`IntentMatrixFactory` を使えば、行列を手動で設計する必要はありません。カテゴリごとのサンプルベクトルを渡すだけで、最適な変換行列を対照学習で自動生成します。

```bash
# ML パッケージも追加
bun add @warpvector/ml
```

```typescript
import { IntentAdapter, cosineSimilarity, normalize } from 'warpvector';
import { IntentMatrixFactory } from 'warpvector/ml';

// 1. サンプルベクトルを用意（実際には embedding モデルの出力を使用）
const techVectors: Float32Array[] = [/* tech ドメインのベクトル群 */];
const bizVectors: Float32Array[] = [/* business ドメインのベクトル群 */];

// 2. ファクトリにカテゴリを登録
const factory = new IntentMatrixFactory(1536); // 次元数を指定
factory.addCategory("tech", techVectors);
factory.addCategory("business", bizVectors);

// 3. Intent行列を自動生成（InfoNCE対照学習）
const intents = await factory.build();

// 4. IntentAdapter に登録して使用
const adapter = new IntentAdapter(1536);
adapter.addIntent("tech", intents.tech);
adapter.addIntent("business", intents.business);

// 5. クエリベクトルを変換
const warpedQuery = adapter.tune(queryVector, "tech");

// 6. 検索結果の取得（変換後のベクトルでDBに問い合わせる）
const similarity = cosineSimilarity(warpedQuery, documentVector);
```

> 💡 **完全な動作例**: `bun run examples/auto-intent.ts` で、10個のドキュメントに対する Vanilla vs Intent Warping の検索結果の違いを体験できます。

---

## 3. 手動でのIntent行列設定

行列を自分で設計したい場合は、`WarpPipeline` を使ってベクトル操作を組み立てます。

```typescript
import { WarpPipeline, IntentWeights } from 'warpvector';

const intents: Record<string, IntentWeights> = {
  // 例えば「リスク分析」という意図の変換定義
  riskAnalysis: {
    matrix: [
      [1.2, 0.1, -0.4],
      [-0.1, 1.5, 0.2],
      [0.3, -0.2, 1.1],
    ],
    bias: [0.05, -0.1, 0.2]
  }
};

// 3次元ベクトルを受け取り、Intent変換と量子化を行うパイプラインを構築
const pipeline = new WarpPipeline(3)
  .addIntent(intents)
  .quantize("int8");
```

### ベクトルの変換（ワープ）

AIモデルから出力された標準の埋め込みベクトルを、指定した意図に合わせて変換します。

```typescript
// AIモデルが出力した元のベクトル
const baseVector = [0.15, -0.23, 0.88];

// "riskAnalysis" の意図に合わせてベクトルを変換し、Int8に量子化
const tunedVector = pipeline.run(baseVector, { intent: "riskAnalysis" });

console.log(tunedVector); // Int8Array に変換された新しいベクトル
```

---

## 4. Auto-blending（自動ルーティング）

複数のインテントがある場合、クエリの内容に応じて自動で最適なインテントをブレンドする機能があります。

```typescript
import { IntentAdapter, normalize } from 'warpvector';

const adapter = new IntentAdapter(1536);

// routingVector を持つインテントを登録
adapter.addIntent("tech", {
  matrix: techMatrix,
  bias: techBias,
  routingVector: normalize(techCentroid), // カテゴリの重心ベクトル
});

adapter.addIntent("business", {
  matrix: bizMatrix,
  bias: bizBias,
  routingVector: normalize(bizCentroid),
});

// クエリの内容に応じて自動的にブレンド比率が決定される
const result = adapter.tuneAutoBlended(queryVector);
```

> 💡 `IntentMatrixFactory` で生成したインテントには `routingVector` が自動的に含まれるため、すぐに `tuneAutoBlended()` を使えます。

---

## 5. 正規化と類似度計算

変換後のベクトルは、Pinecone や Qdrant などのベクトルデータベースにそのまま渡すことができますが、コサイン類似度で比較する場合は、変換後に正規化（L2 Normを1にする）を行うと便利です。

```typescript
import { normalize, cosineSimilarity } from 'warpvector';

const normalized = normalize(tunedVector);

// 別のベクトルとの類似度を計算
const queryVector = normalize([0.1, 0.0, 0.9]);
const similarity = cosineSimilarity(normalized, queryVector);
console.log(`類似度: ${similarity}`);
```

---

## 6. 次のステップ

| やりたいこと | ドキュメント |
|-------------|-------------|
| パイプラインをもっと高度に使う | [Advanced Usage](./advanced-usage.ja.md) |
| メモリコストを75〜97%削減 | [量子化と圧縮](./4-quantization.ja.md) |
| WASMで高速MLP推論 | [ニューラルネットワーク](./2-neural-networks.ja.md) |
| 検索空間の偏りを自動補正 | [オンライン等方化](./3-whitening-pca.ja.md) |
| LangChain / Prisma と統合 | [エコシステム統合](./8-integrations.ja.md) |
| Intent行列を自動生成 | [IntentMatrixFactory](./17-intent-matrix-factory.ja.md) |
| エッジで動かす | [Edge Quickstart](./edge-quickstart.ja.md) |

---

## 7. クイックスタート・サンプルの実行

`warpvector` の効果を実際にターミナル上で確認できるサンプルスクリプトを用意しています。

```bash
# 基本的なクイックスタート（3Dベクトルでの直感的デモ）
bun run examples/quickstart.ts

# Intent行列の自動生成デモ（IntentMatrixFactory を使ったE2E例）
bun run examples/auto-intent.ts
```

### `quickstart.ts` の概要

1. **ドキュメントの定義**: 「天気」「経済」「テクノロジー」の3つのサンプルベクトルを用意します。
2. **通常の検索**: クエリに対して最も近いのは「経済」のドキュメントになります。
3. **ワープの適用**: `IntentAdapter` を用いて、ベクトル空間の「Z軸（テクノロジーの特徴量）」を 2.5倍に拡張し、空間全体をZ軸方向にシフトさせるアフィン変換を適用します。
4. **意図を反映した検索**: テクノロジーのドキュメントが検索スコアの最上位に躍り出ます。

### `auto-intent.ts` の概要

1. **10個のドキュメント**（tech 5個 + business 5個）に対して擬似Embeddingを生成
2. **IntentMatrixFactory** でカテゴリサンプルから自動的にIntent行列を学習
3. **Vanilla検索 vs Intent Warping vs Auto-blending** の3つの検索方式を比較
4. Vanilla では Top5 に tech が 2/5 しかなかったのに対し、Intent Warping では **5/5** に改善
