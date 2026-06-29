# 実践ガイド: 自動学習と連合学習のプロジェクト導入

WarpVectorの最大の特徴は、重いLLMの再学習やベクトルデータベースの再インデックスを行うことなく、ユーザーの検索行動から**リアルタイムにベクトル空間を自己最適化できる**点にあります。

このドキュメントでは、数学的な背景（アフィン変換やLoRA）を踏まえつつ、**既存のプロジェクトを壊さずに最も簡単に自動学習ループと連合学習を組み込む方法**を解説します。

---

## 1. 背後にある技術的アルゴリズム

WarpVectorの「意図（Intent）による空間の歪み」は、以下の数学的・技術的基盤の上に成り立っています。

- **アフィン写像と特徴の交差:** $x' = \sigma(W \cdot x + b)$ の計算により、単なる平行移動ではなく、空間の回転や特徴同士の掛け合わせを実現します。
- **LoRA (Low-Rank Adaptation):** 1536次元のような高次元行列を低ランク（例: ランク8）の2つの行列に分解し、パラメータ数と計算量を **約1/100** に削減しています。
- **WASMとフラットメモリ:** 行列演算は TypeScript ではなく WebAssembly の連続メモリ上で実行され、ガベージコレクションの停止をゼロに抑えた超低遅延計算を実現しています。

これらの軽量化により、Node.js やエッジ環境（Cloudflare Workers等）単体で、**GPUなしの対照学習（InfoNCE / Triplet Loss）** が可能になっています。

---

## 2. 既存プロジェクトへの最も簡単な導入アーキテクチャ

既存の検索システム（RAGやEC検索）を壊さずに導入するには、以下の3ステップの構成が最も安全で効果的です。

### ステップ1: 検索パイプラインへの組み込み（影響ゼロからのスタート）

既存のベクトル検索コードの直前に、WarpVector の `IntentAdapter` を **1行挟むだけ** にします。最初は初期状態（単位行列）なので、検索結果は一切変化しません。

```typescript
import { IntentAdapter } from "warpvector";

const dimension = 1536;
const adapter = new IntentAdapter(dimension);

// 起動時にKVストアなどから最新の重みをロードしてセットしておく
const weights = await kvStore.get("my_domain_weights") || adapter.getIdentityWeights();
adapter.addIntent("search", weights);

// --- 検索実行時 ---
// 既存の埋め込みベクトルをここでワープさせる
const optimizedQuery = adapter.tune(rawQueryVector, "search");

// optimizedQuery を使ってベクトルDB（Pineconeやpgvector）へ投げる
```

### ステップ2: コールドスタート（運用開始前）の初期学習

まだユーザーのログがない状態でも、ある程度の意図を反映させたい場合は `IntentMatrixFactory` を使います。カテゴリごとに少数のサンプル文書を渡すだけで、自動で対照学習が行われ、初期の変換行列が生成されます。

```typescript
import { IntentMatrixFactory } from "warpvector/ml";

const factory = new IntentMatrixFactory(1536);

// 用意したドキュメントのベクトルを5〜10件ずつ渡すだけ
factory.addCategory("search", [ docVec1, docVec2, docVec3, ... ]);

// 内部のAdamオプティマイザが最適な初期行列を数秒で構築
const intents = await factory.build();
await kvStore.put("my_domain_weights", intents["search"]);
```

### ステップ3: フィードバック用APIの追加（運用開始後）

フロントエンドからのクリックログを受け取る専用のエンドポイントを作成し、裏側でひっそりとログを貯め、バッチサイズに達したら学習を発火させます。

```typescript
import { FeedbackCollector, AdaptiveScheduler, InfoNCETrainer } from "warpvector/ml";

const collector = new FeedbackCollector({ dwellThresholdMs: 3000 });
const trainer = new InfoNCETrainer(1536);
// 10件クリックが溜まるたびに自動で学習を発火
const scheduler = new AdaptiveScheduler(trainer, { batchSize: 10, initialLearningRate: 0.01 });

app.post("/api/feedback", async (req, res) => {
  const { queryVector, resultVectors, clickedIndex } = req.body;

  // 1. ログをCollectorに記録
  const impId = collector.recordImpression({ queryVector, resultVectors, timestamp: Date.now() });
  collector.recordFeedback({ impressionId, resultIndex: clickedIndex, type: "click" });

  // 2. 学習データに変換
  const examples = collector.toInfoNCEExamples();
  if (examples.length === 0) return res.send("OK");

  // 3. 学習の発火（バッチサイズに達していれば更新される）
  let currentWeights = await kvStore.get("my_domain_weights");
  const updatedWeights = await scheduler.addFeedback(currentWeights, examples);

  if (updatedWeights) {
    // 4. 更新されたら保存し直す。次回から検索空間が賢くなる。
    await kvStore.put("my_domain_weights", updatedWeights);
    collector.flush();
  }

  res.send("OK");
});
```

---

## 3. 分散環境での安全な重み統合 (Federated Learning)

「日本エッジ」と「米国エッジ」のように複数のサーバーで別々に学習が走っている場合、それぞれの行列がバラバラになってしまいます。
`FederatedAggregator` を使えば、**ユーザーの生の検索ログを中央に送ることなく（プライバシー保護）、学習済みの行列数値（Weights）だけを中央で安全に統合**できます。

```typescript
import { FederatedAggregator } from "warpvector/train";

// 中央の集約サーバーで定期実行するバッチ処理
const aggregator = new FederatedAggregator(globalBaseWeights, 1536);

// 各エッジから送られてきた行列を登録（クリック数が多いほど重視される）
aggregator.submitUpdate({ weights: weightsFromJapan, interactionCount: 150 });
aggregator.submitUpdate({ weights: weightsFromUS, interactionCount: 50 });

// FedAvg (加重平均) で安全にマージ
const newGlobalWeights = aggregator.aggregate();

// 全エッジへ再配布
await distributeToAllEdges(newGlobalWeights);
```

---

## 4. パラメータチューニング・ガイド

自動学習をコントロールする主なパラメータと、空間への影響について解説します。

| パラメータ | 設定値の目安 | 影響と効果 |
|-----------|------------|-----------|
| **`batchSize`** <br/>*(AdaptiveScheduler)* | **5 〜 10** | **リアルタイム性重視:** ユーザーの反応がすぐに空間に反映されます。トレンド変化の激しいEC検索などに適しています。 |
| | **32 〜 64** | **安定性重視:** 外れ値（誤クリックなど）のノイズに強くなり、空間全体が安定して最適化されます。 |
| **`temperature`** <br/>*(InfoNCETrainer)* | **0.05 〜 0.07** | **シャープな空間:** 正解と不正解の違いを極端に区別しようとします。似たような文書から正解だけを鋭く引き寄せますが、過学習に注意が必要です。 |
| | **0.1 〜 0.2** | **マイルドな空間:** 確率分布が滑らかになり、空間全体をじんわりと動かします。一般的な用途での推奨値です。 |

---

このガイドの通りに実装を進めることで、**「最初は既存の検索精度そのままに、運用を続けるにつれてユーザーの行動を学習し、自動的に検索精度が向上していく」** 理想的なベクトル検索システムを低コストで構築することができます。
