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

まず、それぞれの「意図（コンテキスト）」に対応する変換行列とバイアスを用意し、アダプターを初期化します。

```typescript
import { IntentAdapter, IntentWeights } from 'warpvector';

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

const adapter = new IntentAdapter(intents);
```

### ベクトルの変換（ワープ）

AIモデルから出力された標準の埋め込みベクトルを、指定した意図に合わせて変換します。

```typescript
// AIモデルが出力した元のベクトル
const baseVector = [0.15, -0.23, 0.88];

// "riskAnalysis" の意図に合わせてベクトルを変換
const tunedVector = adapter.tune(baseVector, "riskAnalysis");

console.log(tunedVector); // Float32Array に変換された新しいベクトル
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
