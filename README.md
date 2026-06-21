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

`warpvector` は、**「LLMモデルを取り替えたり再学習したりすることなく、検索結果を劇的に賢く・軽く・パーソナライズできる魔法のフィルター」** として機能する次世代DBミドルウェアです。ベースとなるベクトルデータはそのままに、検索の瞬間に軽量な行列演算を適用することで、ファイルやデータ間の意味的類似性をユーザーの「真の意図」に極限まで近づけます。

---

## 🎯 5つの強力なユースケース（具体的に何ができるか？）

`warpvector` を既存の RAG やベクトル検索システムに組み込むことで、以下の課題を解決できます。

### 1. ユーザーの「意図」に合わせたパーソナライズ検索
標準的な埋め込みモデル（OpenAI ada-002 など）は「Apple」が果物か企業かを判別できません。WarpVectorを使えば、「ITドメイン」「食品ドメイン」といった**意図（インテント）**を切り替えるだけで、一瞬でベクトル空間が歪み、クエリベクトルが目的のドキュメントに近づきます。

### 2. 「クリックログ」からエッジでリアルタイム自己学習
検索結果の改善のためにLLMを再学習する必要はありません。ユーザーが「この結果をクリックした（正解）」「これはスルーした（不正解）」というログを、Cloudflare Workers などのエッジサーバー上で直接オンライン学習させることができます。モデル本体はいじらずに、**ユーザー層の好むベクトル変換行列だけをリアルタイムに更新**します。

### 3. モデル特有の「検索空間の偏り」の自動補正
多くの埋め込みモデルは、どんな単語を入れても類似度が高く出てしまう「異方性（空間の偏り）」を抱えています。`WhiteningAdapter` を挟むだけで、流れてくる検索クエリから無駄に偏っている方向をオンラインで自動学習し、その偏りを差し引いて**検索の解像度を劇的に引き上げます。**

### 4. ベクトルDBのメモリコストを 1/4 〜 1/32 に激減させる
1536次元のベクトルを数百万件保存すると莫大なクラウドインフラ料金がかかります。`WarpPipeline` の最後に `.quantize("int8")` を追加するだけで、**精度をほぼ落とさずに（相関 0.9999 以上）データサイズを圧縮**し、DB側での超高速検索が可能になります。

### 5. 現在の TypeScript コードを壊さずに「数行」で導入
Python や重い機械学習ライブラリに依存していると、Node.js やエッジへの組み込みは困難です。WarpVectorは完全な TypeScript ネイティブ ＆ WASM 実装であり、**LangChain、LlamaIndex、Prisma (pgvector)** の設定コードをラッパーでくるむだけで導入が完了します。

---

## 🚀 主な特徴

- **統一された `WarpAdapter` インターフェース [✨ NEW]:** 全てのアダプターが同一の `tune()` メソッドを備えたインターフェースを実装。設計の共通化により、Prisma拡張やLangChain拡張にどのアダプターでもシームレスに組み込めます。
- **次世代DBミドルウェア:** 既存のベクトルDB（Pinecone, Qdrant, DuckDBなど）とフロントエンドの間に立ち、動的なコンテキストルーティングを提供。
- **動的アフィン変換 & 非線形MLP [NEW]:** 単純な行列変換（$W \cdot x + b$）に加え、WASMを用いた超高速な多層パーセプトロン(MLP)と非線形活性化関数(ReLU, Sigmoid, Tanh)による高度な空間変形をサポート。
- **オンライン等方化 (Whitening) [NEW]:** Oja's Rule を用いたオンラインPCAにより、OpenAI `ada-002` などが抱える「検索空間の極端な偏り（異方性）」をエッジ側でストリーミング補正し、検索精度を劇的に向上。
- **ColBERT / Late Interaction (WASM) [✨ NEW]:** 単一ベクトルの代わりに「トークン行列」を用いて検索する最高峰の手法 (ColBERT) を WASM 化。TS環境では絶望的に遅い MaxSim 演算を爆速で処理し、RAGの検索品質を極限まで引き上げます。
- **Vector Quantization (量子化) [✨ NEW]:** `Float32` のベクトルを `Int8` (スカラー量子化: メモリ1/4) または `Binary` (二値化: メモリ1/32) に圧縮。大規模ベクトルデータの保持コストを激減させ、ハミング距離計算で超高速検索を実現します。
- **Hybrid Search (RRF & RSF) [✨ NEW]:** ベクトル検索（Dense）とキーワード検索（Sparse/BM25）の結果を統合するハイブリッド検索ユーティリティ（Reciprocal Rank Fusionなど）を内蔵。
- **WASM/SIMDによる超高速処理:** 行列変換、PCA更新、ColBERT処理にはAssemblyScriptでコンパイルされたインラインWebAssembly（WASM）バックエンドを呼び出し、計算速度を最大化します。
- **InfoNCE & Triplet 学習エンジン (Adam Optimizer内蔵) [NEW]:** Pythonサーバー不要。ユーザーのフィードバックから Contrastive Learning (複数Negative対応の対照学習) をエッジワーカー上で直接オンライン学習。
- **LoRA (低ランク適応) アーキテクチャ:** `LoraIntentAdapter` により、超高次元ベクトル（1536次元など）でもメモリ使用量・計算量を劇的に削減。
- **Prisma + pgvector ネイティブ統合拡張 [NEW]:** Prisma Client Extensionとして透過的に統合。複雑なSQLを書かずに WarpVector で推論・補正されたベクトルでのデータベース検索がメソッド1つで完結。
- **次元削減 / 射影変換 (ProjectionAdapter):** PCAやSVDで事前計算した射影行列を用いたベクトルの次元削減・拡張をWASM高速処理でサポート。
- **タスクベクトル演算 (Task Arithmetic) [✨ NEW]:** 複数の学習済みアダプタの重みを「タスクベクトル」として加減算し、推論時にゼロオーバーヘッドの静的マージを実現。
- **超次元計算 / VSA (Vector Symbolic Architecture) [✨ NEW]:** ベクトルのバインド（結合）・バンドル（重ね合わせ）・アンバインド（抽出）を提供。Binary VSA（XOR演算）にも対応し、メタデータをベクトルに埋め込んだまま検索が可能。
- **TypeScriptネイティブ & ゼロ依存:** 外部の機械学習ライブラリは一切不要。Cloudflare Workers、Bun、Node.jsなどのモダンなサーバーレス/エッジ環境に完全対応。

---

## 基本的な使い方 (WarpPipeline)

新しく導入された `WarpPipeline` を使うと、複雑なベクトル変換（非線形推論、インテント変換、量子化）からDBフォーマットへの出力までを数行で直感的に記述できます。

```typescript
import { WarpPipeline } from 'warpvector';
import { MlpAdapter } from 'warpvector/ml';
import { QuantizationAdapter } from 'warpvector/extras';

// MLP アダプタと量子化アダプタを事前に作成
const mlp = new MlpAdapter(layers);
const quantizer = new QuantizationAdapter({ type: "int8", dim: 1536 });

// 1. パイプラインの構築
const pipeline = new WarpPipeline(1536)
  .addStep("MlpAdapter", mlp)                 // 非線形変換 (WASM使用)
  .addIntent({ "domain_x": intentWeights })   // ユーザーごとに空間を歪める
  .setFinalStage("QuantizationAdapter", quantizer); // 最後に Int8 に量子化して圧縮

// 2. 非同期初期化 (WASMモジュールのセットアップなどを一括実行)
await pipeline.init();

// 3. 超高速バッチ処理 (WASM/SIMD対応)
const batchVectors = [[0.1, ...], [0.5, ...]]; // 大量データ
const results = pipeline.runBatch(batchVectors, { intent: "domain_x" });

// 4. Vector DB 用フォーマットへの直接出力
const pineconeQuery = pipeline.runAndFormat(
  rawVector, 
  { format: "pinecone", topK: 10, filter: { genre: "action" } }
);

// 5. パイプライン丸ごとの永続化と復元
const stateJson = pipeline.exportState(); 
const restoredPipeline = WarpPipeline.importState(stateJson);
```

## 各機能のドキュメント (Documentation)

各機能のより詳細な仕組み、コードスニペット、ユースケースについては以下の個別ドキュメントをご参照ください。

1. **[コアアダプタ (Core Adapters)](./docs/1-core-adapters.md)**
   - `IntentAdapter`, `ProjectionAdapter`, `LoraIntentAdapter` の基本
2. **[ニューラルネットワーク (Neural Networks)](./docs/2-neural-networks.md)**
   - `MlpAdapter` を用いた多層パーセプトロンと非線形活性化関数
3. **[オンライン等方化・PCA (Whitening)](./docs/3-whitening-pca.md)**
   - 空間的偏り (異方性) のストリーミング学習による除去
4. **[量子化と圧縮 (Quantization)](./docs/4-quantization.md)**
   - `Int8` (1/4圧縮) および `Binary` (1/32圧縮) による高速化と省メモリ化
5. **[Late Interaction / ColBERT](./docs/5-colbert.md)**
   - WASM を用いた MaxSim 演算による緻密なトークン照合
6. **[ハイブリッド検索フュージョン (Hybrid Search)](./docs/6-hybrid-search.md)**
   - ベクトル検索とキーワード検索の統合 (`RRF`, `RSF`)
7. **[オンライン学習エンジン (Trainers)](./docs/7-trainers.md)**
   - 対照学習によるリアルタイムな空間最適化 (`InfoNCETrainer`, `TripletTrainer`)
8. **[エコシステム統合 (Integrations)](./docs/8-integrations.md)**
   - `LangChain`, `LlamaIndex`, `Prisma + pgvector` とのシームレスな連携
9. **[状態の永続化・シリアライズ (Serialization)](./docs/9-serialization.md)**
   - 学習結果の JSON / バイナリ形式での保存と復元
10. **[次元削減・モデル間移行 (Projection & Migration)](./docs/10-projection-migration.md)**
    - `ProjectionAdapter` による射影変換と `MigrationTrainer` によるモデル間移行
11. **[タスクベクトル演算 (Task Arithmetic)](./docs/11-task-arithmetic.md)**
    - 学習済み重みの加減算によるゼロオーバーヘッドのモデルマージ
12. **[超次元計算 / VSA (Vector Symbolic Architecture)](./docs/12-vsa.md)**
    - ベクトルのバインド・バンドル・アンバインドによるメタデータ埋め込み演算

---

## 📦 インストール

```bash
npm install warpvector
# または
bun add warpvector
```

コア機能（IntentAdapter, MlpAdapter, WhiteningAdapter, 各Trainer, 量子化, VSA 等）は**ゼロ依存**で動作します。

Prisma や LangChain との統合機能を使う場合は、それぞれの依存を追加でインストールしてください：

```bash
# Prisma 統合（pgvector）
npm install @prisma/client sql-template-tag

# LangChain / LlamaIndex 統合
npm install @langchain/core
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
  { matrix: matrix1, bias: bias1, activation: "relu" },   // 1536 -> 128
  { matrix: matrix2, bias: bias2, activation: "linear" }  // 128 -> 2
]);

// WASM の初期化 (重みもWASMメモリに永続化される)
await mlp.init();

// 超高速非線形推論 (WASM)
const output = mlp.tune(baseVector);
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

### 5. 高品質 RAG のための Late Interaction (ColBERT) [✨ NEW]

単一ベクトルでは潰れてしまう細かいニュアンスを、トークンごとの行列で保持し、WASMの超高速 MaxSim 演算によって緻密に照合します。

```typescript
import { ColbertAdapter } from 'warpvector';

const adapter = new ColbertAdapter();

// queryTokens: クエリのトークン行列 (平坦化された Float32Array)
// documentTokensArray: 各ドキュメントのトークン行列の配列
const results = adapter.rank(queryTokens, [doc1Tokens, doc2Tokens, doc3Tokens], 1536);

console.log(results); // [{ index: 1, score: 1.44 }, { index: 0, score: 0.76 }, ...]
```

### 6. Hybrid Search / 検索結果の統合 (RRF & RSF) [✨ NEW]

ベクトル検索結果（Dense）とキーワード検索結果（Sparse/BM25）をシームレスに統合（フュージョン）するための独立したアルゴリズムを提供します。WarpVector のコアなベクトル変形機能と組み合わせることで、最高峰の検索精度を達成できます。

```typescript
import { rrf, rsf } from 'warpvector';

const denseResults = [
  { id: "doc1", score: 0.95 }, 
  { id: "doc2", score: 0.88 }
];

const sparseResults = [
  { id: "doc2", score: 15.2 }, 
  { id: "doc1", score: 12.1 }
];

// RRF (Reciprocal Rank Fusion): スコアの絶対値に依存せず、順位(Rank)のみを使って公平に統合
const rrfResults = rrf([denseResults, sparseResults]);

// RSF (Relative Score Fusion): Min-Max正規化を行い、重み(Dense 70%, Sparse 30%)をつけて加算統合
const rsfResults = rsf([denseResults, sparseResults], [0.7, 0.3]);
```

### 7. ベクトル量子化 (Vector Quantization) [✨ NEW]

メモリ制約の厳しいエッジ環境向けに、`Float32` (32ビット) のベクトルを `Int8` (8ビットスカラー) または `Binary` (1ビット) に圧縮するアダプターを提供します。

```typescript
import { QuantizationAdapter } from 'warpvector';

// Int8 量子化 (メモリを 1/4 に削減)
const int8Adapter = new QuantizationAdapter({ type: "int8", dim: 1536 });
const int8Vec = int8Adapter.tune(floatVector); // Int8Array
const dot = QuantizationAdapter.int8DotProduct(int8Vec, int8Vec2);

// Binary 量子化 (メモリを 1/32 に削減)
const binaryAdapter = new QuantizationAdapter({ type: "binary", dim: 1536 });
const binVec = binaryAdapter.tune(floatVector); // Uint8Array(192バイト)
const dist = QuantizationAdapter.hammingDistance(binVec, binVec2); // 超高速なXORハミング距離計算
```

### 8. 動的学習エンジン (Trainers with Adam) [✨ UPGRADED]

Pythonサーバーを立てることなく、ユーザーのフィードバックをもとにエッジ上でベクトル空間を最適化できます。Adamオプティマイザーと InfoNCE Loss (複数Negative) に対応しました。

```typescript
import { InfoNCETrainer } from 'warpvector';

const trainer = new InfoNCETrainer(1536);

// 1つの正解と複数の不正解（In-batch Negatives）を同時に学習
const updatedWeights = await trainer.updateOnline(
  currentWeights,
  {
    anchor: anchorVector,
    positive: positiveVector,
    negatives: [negativeVector1, negativeVector2],
  },
  { learningRate: 0.001, temperature: 0.1 }
);
```

### 9. LangChain / LlamaIndex との統合 (Integrations)

`warpvector` は、既存の巨大エコシステムに「たった数行」で組み込むことができます。

#### LangChain 統合 (`WarpEmbeddings`)
```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { IntentAdapter } from "warpvector";
import { WarpEmbeddings } from "warpvector/integrations/langchain";

const baseEmbeddings = new OpenAIEmbeddings();
const adapter = new IntentAdapter(myIntents);

const warpEmbeddings = new WarpEmbeddings({
  baseEmbeddings, adapter, intentName: "riskAnalysis"
});
// 検索時のみ動的ワープが適用されます
const vectorStore = new MemoryVectorStore(warpEmbeddings);
```

#### LlamaIndex 統合 (`WarpLlamaIndexEmbeddings`) [✨ NEW]
```typescript
import { OpenAIEmbedding } from "llamaindex";
import { WarpLlamaIndexEmbeddings } from "warpvector/integrations/llama-index";

const warpLlamaIndexEmbeddings = new WarpLlamaIndexEmbeddings({
  baseEmbeddings: new OpenAIEmbedding(),
  adapter: intentAdapter,
  intentName: "riskAnalysis"
});
// LlamaIndex の VectorStoreIndex などに直接渡せます
```

### 10. 全アダプタの状態永続化 (Universal Serialization) [✨ NEW]

Cloudflare Workers 等の揮発性環境でも、オンライン学習やPCAで得られたコンポーネントを即座にJSONで保存・復元できます。

```typescript
import { WhiteningAdapter } from 'warpvector';

const adapter = new WhiteningAdapter(1536);
// ... オンライン学習 (update) を実行 ...

// 状態をシリアライズしてRedis等に保存
const stateJson = adapter.exportState();

// 次回起動時や別インスタンスで即座に復元
const restoredAdapter = WhiteningAdapter.importState(stateJson);
```

### 11. 次元削減・拡張 (ProjectionAdapter)

PCAやSVDで事前計算した射影行列を用いて、ベクトルの次元数を変換します。WASMによる高速処理にも対応しています。

```typescript
import { ProjectionAdapter } from 'warpvector';

// 1536次元から512次元への射影行列を設定
const adapter = new ProjectionAdapter(1536, 512, {
  v1: { matrix: projectionMatrix, bias: projectionBias }
});

// 次元削減を実行 (WASM使用)
const compressedVector = adapter.tune(baseVector, "v1"); // 512次元
```

### 12. モデル間移行トレーナー (MigrationTrainer) [✨ NEW]

埋め込みモデルを変更する際（例: `ada-002` → `text-embedding-3-small`）に、既存のベクトルを新モデルの空間に翻訳する射影行列を自動学習します。

```typescript
import { MigrationTrainer } from 'warpvector';

// 旧モデル(1536次元)から新モデル(512次元)への翻訳行列を学習
const trainer = new MigrationTrainer(1536, 512);

// 同一テキストを新旧モデルで埋め込み、ペアとして学習データに追加
trainer.addExample({ source: adaVector, target: v3SmallVector });
trainer.addExample({ source: adaVector2, target: v3SmallVector2 });

// 射影行列を学習 (Adam Optimizer)
const projectionWeights = await trainer.train({ epochs: 200, autoTune: true });
```

### 13. タスクベクトル演算 (Task Arithmetic) [✨ NEW]

複数の学習済みアダプタ重みを「タスクベクトル」として加減算し、新しいアダプタを推論時ゼロオーバーヘッドで合成できます。

```typescript
import { TaskArithmetic } from 'warpvector';

// 「法律ドメイン」と「金融ドメイン」の学習済み重みをマージ
const mergedWeights = TaskArithmetic.merge([
  { weights: legalWeights, scale: 0.7 },   // 法律を70%
  { weights: financeWeights, scale: 0.3 },  // 金融を30%
]);

// マージされた重みは通常の IntentWeights として即座に使用可能
adapter.addIntent("legal_finance", mergedWeights);
```

### 14. 超次元計算 / VSA (VsaAdapter) [✨ NEW]

ベクトル・シンボリック・アーキテクチャ（VSA）により、キーと値の概念を1つの密ベクトルに埋め込み、検索空間上でそのまま演算できます。

```typescript
import { VsaAdapter } from 'warpvector';

// バンドル（重ね合わせ）: 複数の概念を1つのベクトルに統合
const bundled = VsaAdapter.bundle([scienceVec, technologyVec]);

// バインド（結合）: キーと値をアダマール積で結合
const bound = VsaAdapter.bind(userIdVec, preferenceVec);

// アンバインド（抽出）: キーを使って値を取り出す
const recovered = VsaAdapter.unbind(bound, userIdVec);

// Binary VSA: 量子化ベクトルに対するXOR演算による超高速処理
const binaryBound = VsaAdapter.bindBinary(binKey, binValue);
const binaryRecovered = VsaAdapter.unbindBinary(binaryBound, binKey);
```

### 15. ベクトルユーティリティ (Slerp / Reject)

高次元空間での幾何学的操作のためのユーティリティ関数を提供します。

```typescript
import { slerp, reject } from 'warpvector';

// 球面線形補間 (Slerp): コサイン類似度を保ちながらベクトル間を滑らかに補間
const interpolated = slerp(vectorA, vectorB, 0.3); // A寄り30%の中間点

// 直交射影 (Reject / Negative Prompting): 特定の概念を完全に除去
// 例: 検索クエリから「政治」の方向成分を取り除く
const filteredQuery = reject(searchVector, politicsVector);
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
