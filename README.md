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

`warpvector` は、次世代のベクトルデータベースのミドルウェアとして機能します。ベースとなるベクトルデータはそのままに、検索の瞬間に数KBの軽量な「意図行列（Matrix）」と「バイアス（Bias）」を適用することで、ベクトル空間全体をインメモリで高速に変形（アフィン変換）させます。これにより、ファイルやデータ間の意味的類似性をユーザーの「真の意図」に極限まで近づけ、次世代の検索体験を提供します。

---

## 🚀 主な特徴

- **次世代DBミドルウェア:** 既存のベクトルDB（Pinecone, Qdrant, DuckDBなど）とフロントエンドの間に立ち、動的なコンテキストルーティングを提供。
- **動的アフィン変換 ($W \cdot x + b$):** 回転・拡大・縮小（行列 $W$）と平行移動（バイアス $b$）を組み合わせ、コンテキストに最適化されたベクトル空間を生成。
- **WASM/SIMDによる超高速バッチ処理:** 大量のベクトル処理にはAssemblyScriptでコンパイルされたインラインWebAssembly（WASM）バックエンドを自動的に呼び出し、計算速度を最大化します。
- **意図の動的合成（Blending）と自動ルーティング:** `tuneBlended` による手動合成に加え、自己アテンション型の `tuneAutoBlended` を搭載。
- **LoRA (低ランク適応) アーキテクチャ:** `LoraIntentAdapter` により、超高次元ベクトル（1536次元など）でもメモリ使用量・計算量を劇的に削減。
- **SGD + Momentum トレーニング内蔵:** `IntentTrainer` や `MigrationTrainer` を使い、ブラウザやエッジ環境上で動的にベクトル行列を学習・最適化可能。
- **次元圧縮・プロジェクション:** `ProjectionAdapter` を使った、1536次元から512次元などへのセマンティックな次元削減機能。
- **TypeScriptネイティブ & ゼロ依存:** Python環境や重厚な機械学習ライブラリは一切不要。
- **エッジ・ローカルファースト対応:** Cloudflare Workers、Bun、Node.jsなどのモダンなサーバーレス/エッジ環境に即座に組み込み可能。

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
    bias: [0.05, -0.1, 0.2],
    routingVector: [1.0, 0.0, 0.0] // 自動ブレンド計算用の代表方向
  }
};

const adapter = new IntentAdapter(myIntents);
const baseVector = [0.15, -0.23, 0.88];

// "riskAnalysis" の意図に合わせてベクトルをワープ (ReLU活性化も可能)
const warpedVector = adapter.tune(baseVector, "riskAnalysis", "relu");
```

### 2. LoRAによる高次元ベクトルの高速変換 (LoraIntentAdapter)

1536次元などの高次元モデル（例: OpenAI `text-embedding-3-small`）の変換には、LoRAを使用することでメモリ消費と計算コストを大幅に削減できます。

```typescript
import { LoraIntentAdapter } from 'warpvector';

// 次元数 1536, ランク数 16 で初期化
const loraAdapter = new LoraIntentAdapter(1536, 16);

// 意図ごとの低ランク行列 (AとB) を追加
loraAdapter.addIntent("scientific", {
  matrixA: [...], // 16 x 1536
  matrixB: [...], // 1536 x 16
  bias: [...]     // 1536
});

const warpedVector = loraAdapter.tune(baseVector, "scientific");
```

### 3. 動的学習エンジン (Trainers)

Pythonサーバーを立てることなく、ユーザーのフィードバック（クリックやいいね）をもとに、フロントエンドやエッジワーカー上で動的に行列を最適化（SGD + Momentum）できます。

```typescript
import { IntentTrainer, MigrationTrainer } from 'warpvector';

// 学習率 0.01、モメンタム 0.9 でトレーナーを初期化
const trainer = new IntentTrainer(1536, {
  learningRate: 0.01,
  momentum: 0.9,
  batchSize: 32
});

// オンライン学習 (1データごとの逐次学習)
const loss = trainer.updateOnline(
  inputVector,        // ベースとなるベクトル
  targetVector,       // 理想とするベクトル
  currentWeights      // 現在の行列・バイアス
);

console.log(`Current Loss: ${loss}`);
```

### 4. 次元のマイグレーション (ProjectionAdapter)

古い次元数のモデル（例: 1536次元）から新しい次元数のモデル（例: 512次元）へ、セマンティクスを保持したままベクトル空間を投影（マイグレーション）します。

```typescript
import { ProjectionAdapter } from 'warpvector';

const projectionWeights = {
  matrix: [...], // 512 x 1536 行列
  bias: [...]    // 512 バイアス
};

// 入力1536次元、出力512次元のアダプターを初期化
const projector = new ProjectionAdapter(1536, 512, { "v1_to_v2": projectionWeights });
const reducedVector = projector.project(highDimVector, "v1_to_v2"); // 1536 -> 512
```

### 5. LangChain / LlamaIndex との統合 (Integrations)

`warpvector` は、LangChain などの既存のエコシステムに「たった数行」で組み込むことができます。
`WarpEmbeddings` クラスを使用することで、クエリ検索時のみベクトル空間を動的にワープさせ、Pinecone や Qdrant などのあらゆる VectorStore にそのまま渡すことができます。

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { IntentAdapter } from "warpvector";
import { WarpEmbeddings } from "warpvector/integrations/langchain";

// 1. 通常の Embeddings と IntentAdapter を初期化
const baseEmbeddings = new OpenAIEmbeddings();
const adapter = new IntentAdapter(myIntents);

// 2. ラップする！
const warpEmbeddings = new WarpEmbeddings({
  baseEmbeddings,
  adapter,
  intentName: "riskAnalysis" // 動的に変更可能
});

// 3. あとは通常の LangChain のワークフローに渡すだけ
const vectorStore = new MemoryVectorStore(warpEmbeddings);

// 実行時に意図を切り替える場合
warpEmbeddings.setIntent("economicImpact");
const results = await vectorStore.similaritySearch("Market crash", 5);
```

---

## 📐 数学的背景：動的アフィン変換と非線形性

入力となる標準的なベースベクトル $\mathbf{x} \in \mathbb{R}^d$ に対し、`warpvector` は以下の**アフィン写像（Affine Map）**を適用し、調律された新しいベクトル $\mathbf{x}' \in \mathbb{R}^d$ を生成します。

$$\mathbf{x}' = \sigma(\mathbf{W}_I \mathbf{x} + \mathbf{b}_I)$$

- $\mathbf{W}_I \in \mathbb{R}^{d \times d}$ ：**意図変換行列（Intent Matrix）**。空間の回転や特徴量の強調（歪み）を担当します。
- $\mathbf{b}_I \in \mathbb{R}^d$ ：**意図バイアスベクトル（Intent Bias）**。空間全体を特定のコンテキストへ平行移動（シフト）させます。
- $\sigma$ ：**非線形活性化関数（Activation Function）**。空間を [0, 1] や [-1, 1] などへ曲げ込み、複雑な意味の切り分けを可能にします (`relu`, `sigmoid`, `tanh`)。

この計算複雑度はわずか $\mathcal{O}(d^2)$ （LoRAの場合は $\mathcal{O}(d \cdot r)$）であり、埋め込みモデルを再実行するのに比べて圧倒的に高速です。
WASM（WebAssembly）と `Float32Array` によるメモリアライメント最適化を活用することで、**ブラウザ上やエッジ環境でも数千〜数万件のバッチ処理を数ミリ秒で完了**させることができます。

---

## 🤝 貢献 (Contributing)

本プロジェクトはオープンソースです。新機能の追加、VectorDB用の最適化アダプターの提供、パフォーマンス改善のプルリクエストを歓迎します！

## 📄 ライセンス

MIT License
