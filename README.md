# warpvector 🌌

[![npm version](https://badge.fury.io/js/warpvector.svg)](https://badge.fury.io/js/warpvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge Ready](https://img.shields.io/badge/Edge-Ready-success.svg)](#)
[![Zero Dependency](https://img.shields.io/badge/Dependencies-0-brightgreen.svg)](#)

`warpvector` は、AIモデルの再学習や重い再推論を行うことなく、検索クエリやコンテキスト（意図）に応じてベクトル空間を動的に変形させる、TypeScriptネイティブの軽量ミドルウェア・ユーティリティです。

データベースとアプリケーションの間に立ち、インメモリの高速なアフィン変換によって、データの意味的距離をユーザーの「真の意図」に極限まで近づけます。

---

## 💡 なぜ `warpvector` なのか？

従来のベクトル検索は静的であり、事前に生成された埋め込みベクトルの距離（類似度）に依存していました。コンテキストに応じた検索の微調整を行いたい場合、これまではメタデータのフィルタリングに頼るか、重い指示チューニング型モデルを再度動かすしかなく、リアルタイム性や柔軟性に欠けていました。

`warpvector` は、次世代のベクトルデータベースのミドルウェアとして機能します。ベースとなるベクトルデータはそのままに、検索の瞬間に数KBの軽量な「意図行列（Matrix）」と「バイアス（Bias）」を適用することで、ベクトル空間全体をインメモリで高速に変形させます。これにより、ファイルやデータ間の意味的類似性をユーザーの「真の意図」に極限まで近づけ、次世代の検索体験を提供します。

---

## 🚀 主な特徴

- **次世代DBミドルウェア:** 既存のベクトルDB（Pinecone, Qdrant, DuckDBなど）とフロントエンドの間に立ち、動的なコンテキストルーティングを提供。
- **動的アフィン変換 & 非線形MLP [NEW]:** 単純な行列変換（$W \cdot x + b$）に加え、WASMを用いた超高速な多層パーセプトロン(MLP)と非線形活性化関数(ReLU, Sigmoid, Tanh)による高度な空間変形をサポート。
- **オンライン等方化 (Whitening) [NEW]:** Oja's Rule を用いたオンラインPCAにより、OpenAI `ada-002` などが抱える「検索空間の極端な偏り（異方性）」をエッジ側でストリーミング補正し、検索精度を劇的に向上。
- **WASM/SIMDによる超高速バッチ処理:** 大量のベクトル処理にはAssemblyScriptでコンパイルされたインラインWebAssembly（WASM）バックエンドを自動的に呼び出し、計算速度を最大化します（130万推論/秒以上）。
- **InfoNCE & Triplet 学習エンジン (Adam Optimizer内蔵) [NEW]:** Pythonサーバー不要。ユーザーのフィードバックから Contrastive Learning (複数Negative対応の対照学習) をエッジワーカー上で直接オンライン学習。
- **LoRA (低ランク適応) アーキテクチャ:** `LoraIntentAdapter` により、超高次元ベクトル（1536次元など）でもメモリ使用量・計算量を劇的に削減。
- **Prisma + pgvector ネイティブ統合拡張 [NEW]:** Prisma Client Extensionとして透過的に統合。複雑なSQLを書かずに WarpVector で推論・補正されたベクトルでのデータベース検索がメソッド1つで完結。
- **TypeScriptネイティブ & ゼロ依存:** 外部の機械学習ライブラリは一切不要。Cloudflare Workers、Bun、Node.jsなどのモダンなサーバーレス/エッジ環境に完全対応。

---

## 📦 インストール

```bash
npm install warpvector
# または
bun add warpvector
```

---

## 🛠 クイックスタート

### 1. 基本的なアフィン変換 (IntentAdapter)

```typescript
import { IntentAdapter } from 'warpvector';

// 意図ごとの変換行列とバイアスを定義
const myIntents = {
  riskAnalysis: {
    matrix: [
      [1.2, 0.1, -0.4],
      [-0.1, 1.5, 0.2],
      [0.3, -0.2, 1.1],
    ],
    bias: [0.05, -0.1, 0.2]
  }
};

const adapter = new IntentAdapter(myIntents);
const baseVector = [0.15, -0.23, 0.88];

// "riskAnalysis" の意図に合わせてベクトルをワープ
const warpedVector = adapter.tune(baseVector, "riskAnalysis");
```

### 2. 多層ニューラルネットワークの高速推論 (MlpAdapter) [✨ NEW]

WASMバックエンドにより、ブラウザやエッジ環境で重厚なフレームワークなしに多層MLPの推論が可能です。

```typescript
import { MlpAdapter } from 'warpvector';

// 1536次元から128次元の中間層を経て2次元に出力する2層MLP
const mlp = new MlpAdapter([
  { inputDim: 1536, outputDim: 128, activation: "relu" },
  { inputDim: 128, outputDim: 2, activation: "linear" }
]);

mlp.setLayerWeights(0, matrix1, bias1);
mlp.setLayerWeights(1, matrix2, bias2);

// 超高速非線形推論 (WASM)
const output = mlp.predict(baseVector);
```

### 3. 検索空間の等方化 (Online Whitening) [✨ NEW]

事前学習済みモデル（ada-002等）特有の「全ての類似度が高く出てしまう空間の偏り」をオンラインで自動補正します。

```typescript
import { WhiteningAdapter } from 'warpvector';

// トップ1つの主成分（偏り）をストリーミング学習して除去するアダプター
const adapter = new WhiteningAdapter(1536, { learningRate: 0.01, numComponents: 1 });

// ベクトルを受信するたびに自動で偏りの方向を学習 (Oja's Rule)
adapter.update(rawVector1);
adapter.update(rawVector2);

// 検索時に偏りを除去（検索の解像度が劇的に向上）
const whitenedVector = adapter.tune(searchVector);
```

### 4. Prisma + pgvector ネイティブ統合 [✨ NEW]

WarpVector を Prisma Client Extension としてアタッチすることで、ベクトル推論とデータベース検索を統合します。

```typescript
import { PrismaClient } from '@prisma/client';
import { withWarpVector } from 'warpvector/integrations/prisma';
import { WhiteningAdapter } from 'warpvector';

const adapter = new WhiteningAdapter(1536);

// Prisma Client に WarpVector 拡張をアタッチ
const prisma = new PrismaClient().$extends(
  withWarpVector({
    adapter: adapter,
    vectorField: "embedding", // DB上の pgvector 保存先カラム名
    distanceOperator: "<=>"   // コサイン距離を使用
  })
);

// 生のベクトルを渡すだけ！（内部でWarpVector推論とpgvector用SQL生成が自動で行われる）
const results = await prisma.document.searchByVector({
  vector: rawSearchVector,
  topK: 10,
  where: "category = 'science'" // 通常のSQL WHERE条件も併用可能
});
```

### 5. 動的学習エンジン (Trainers with Adam) [✨ UPGRADED]

Pythonサーバーを立てることなく、ユーザーのフィードバックをもとにエッジ上でベクトル空間を最適化できます。Adamオプティマイザーと InfoNCE Loss (複数Negative) に対応しました。

```typescript
import { InfoNCETrainer } from 'warpvector';

const trainer = new InfoNCETrainer(1536, { learningRate: 0.001 });

// 1つの正解と複数の不正解（In-batch Negatives）を同時に学習
const loss = trainer.updateOnline(
  anchorVector, 
  positiveVector, 
  [negativeVector1, negativeVector2], 
  currentWeights
);
```

### 6. LangChain / LlamaIndex との統合 (Integrations)

`warpvector` は、LangChain などの既存のエコシステムに「たった数行」で組み込むことができます。
`WarpEmbeddings` クラスを使用することで、クエリ検索時のみベクトル空間を動的にワープさせ、そのまま VectorStore に渡すことができます。

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { IntentAdapter } from "warpvector";
import { WarpEmbeddings } from "warpvector/integrations/langchain";

const baseEmbeddings = new OpenAIEmbeddings();
const adapter = new IntentAdapter(myIntents);

// ベースのEmbeddingsをラップする
const warpEmbeddings = new WarpEmbeddings({
  baseEmbeddings,
  adapter,
  intentName: "riskAnalysis"
});

const vectorStore = new MemoryVectorStore(warpEmbeddings);
const results = await vectorStore.similaritySearch("Market crash", 5);
```

---

## 📐 数学的背景：動的アフィン変換と非線形性

入力となる標準的なベースベクトル $\mathbf{x} \in \mathbb{R}^d$ に対し、`warpvector` は以下の**アフィン写像（Affine Map）**を適用し、調律された新しいベクトル $\mathbf{x}' \in \mathbb{R}^d$ を生成します。

$$\mathbf{x}' = \sigma(\mathbf{W}_I \mathbf{x} + \mathbf{b}_I)$$

- $\mathbf{W}_I \in \mathbb{R}^{d \times d}$ ：**意図変換行列（Intent Matrix）**。空間の回転や特徴量の強調（歪み）を担当します。
- $\mathbf{b}_I \in \mathbb{R}^d$ ：**意図バイアスベクトル（Intent Bias）**。空間全体を特定のコンテキストへ平行移動（シフト）させます。
- $\sigma$ ：**非線形活性化関数（Activation Function）**。空間を曲げ込み、複雑な意味の切り分けを可能にします (`relu`, `sigmoid`, `tanh`)。

この計算複雑度はわずか $\mathcal{O}(d^2)$ （LoRAの場合は $\mathcal{O}(d \cdot r)$）であり、WASM（WebAssembly）と `Float32Array` によるメモリアライメント最適化を活用することで、**ブラウザ上やエッジ環境でも数千〜数万件の推論を数ミリ秒で完了**させることができます。

---

## 🤝 貢献 (Contributing)

本プロジェクトはオープンソースです。新機能の追加、VectorDB用の最適化アダプターの提供、パフォーマンス改善のプルリクエストを歓迎します！

## 📄 ライセンス

MIT License
