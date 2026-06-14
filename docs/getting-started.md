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

## 2. 基本的な使い方

`warpvector` の中心となるのは `IntentAdapter` クラスです。

### アダプターの初期化

まず、それぞれの「意図（コンテキスト）」に対応する変換行列とバイアスを用意し、アダプターを初期化します。ここでは最もシンプルで強力な `WarpPipeline` を使ってベクトル操作を組み立てます。

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

## 3. 正規化と類似度計算

変換後のベクトルは、Pinecone や Qdrant などのベクトルデータベースにそのまま渡すことができますが、コサイン類似度で比較する場合は、変換後に正規化（L2 Normを1にする）を行うと便利です。

```typescript
import { normalize, cosineSimilarity } from 'warpvector';

const normalized = normalize(tunedVector);

// 別のベクトルとの類似度を計算
const queryVector = normalize([0.1, 0.0, 0.9]);
const similarity = cosineSimilarity(normalized, queryVector);
console.log(`類似度: ${similarity}`);
```

さらに高度な使い方（自動ブレンド、WASMバッチ処理など）については、[Advanced Usage](./advanced-usage.md) をご覧ください。

---

## 4. クイックスタート・サンプルの実行

`warpvector` の効果を実際にターミナル上で確認できるサンプルスクリプトを用意しています。このスクリプトでは、あらかじめ定義された3つのドキュメントベクトルに対して、「通常の検索」と「テクノロジー重視にワープ（アフィン変換）した検索」のスコアの違いを体験できます。

リポジトリをクローンした状態で、以下のコマンドを実行してください。

```bash
# Bun をお使いの場合
bun run examples/quickstart.ts

# Node.js をお使いの場合 (ts-node 等が必要です)
npx ts-node examples/quickstart.ts
```

### サンプルコードの概要

`examples/quickstart.ts` では以下のような処理が行われています。
1. **ドキュメントの定義**: 「天気」「経済」「テクノロジー」の3つのサンプルベクトルを用意します。
2. **通常の検索**: クエリに対して最も近いのは「経済」のドキュメントになります。
3. **ワープの適用**: `IntentAdapter` を用いて、ベクトル空間の「Z軸（テクノロジーの特徴量）」を 2.5倍に拡張（拡大）し、さらに空間全体をZ軸方向にシフトさせるアフィン変換を適用します。
4. **意図を反映した検索**: データベース側（ドキュメント側）のベクトルは一切変更していないにも関わらず、検索クエリが「テクノロジー重視空間」に歪められるため、テクノロジーのドキュメントが検索スコアの最上位に躍り出ます。

このように、フロントエンド（またはエッジ層）で `warpvector` を挟むだけで、ユーザーの意図に応じた検索結果のランキング操作をリアルタイムに実現できます！
