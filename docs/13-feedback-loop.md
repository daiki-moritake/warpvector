# §13 フィードバックループ学習

WarpVector は「使えば使うほど賢くなる」フライホイール学習機構を提供します。ユーザーの検索行動（クリック、スキップ、滞在時間）を自動的に学習データに変換し、ベクトル空間をリアルタイムに最適化します。

---

## 概要: 3層の学習ループ

```
┌─────────────────────────────────────────────┐
│       Layer 3: グローバル最適化              │
│   FederatedAggregator で全ユーザーの         │
│   重みを集約 → ベースライン更新              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│       Layer 2: スケジュール制御              │
│   AdaptiveScheduler が学習率を自動減衰       │
│   初期: 大きく変形 → 後期: 微調整           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│       Layer 1: フィードバック収集            │
│   FeedbackCollector がクリック/スキップを     │
│   TripletExample / InfoNCEExample に変換     │
└─────────────────────────────────────────────┘
```

---

## FeedbackCollector

ユーザーの暗黙的フィードバックを収集し、学習データに自動変換します。

### 基本的な使い方

```typescript
import { FeedbackCollector } from "@warpvector/ml";

const collector = new FeedbackCollector({
  dwellThresholdMs: 3000,  // 3秒以上の閲覧 = positive
  maxImpressions: 200,     // 最大保持数
});

// 1. 検索結果を表示した時にインプレッションを記録
const impressionId = collector.recordImpression({
  queryVector: queryEmbedding,      // 検索クエリのベクトル
  resultVectors: [doc1Vec, doc2Vec, doc3Vec], // 表示された結果
  timestamp: Date.now(),
});

// 2. ユーザーのアクションを記録
// クリック
collector.recordFeedback({
  impressionId,
  resultIndex: 0,
  type: "click",
});

// スキップ（表示されたが無視された）
collector.recordFeedback({
  impressionId,
  resultIndex: 2,
  type: "skip",
});

// 滞在時間
collector.recordFeedback({
  impressionId,
  resultIndex: 1,
  type: "dwell",
  value: 5000,  // 5秒間閲覧
});
```

### 学習データへの変換

```typescript
// TripletTrainer 用に変換
const triplets = collector.toTripletExamples();
// → [{ anchor: queryVec, positive: clickedDoc, negative: skippedDoc }, ...]

// InfoNCETrainer 用に変換（複数 negative をまとめる）
const infoNCE = collector.toInfoNCEExamples();
// → [{ anchor: queryVec, positive: clickedDoc, negatives: [skip1, skip2] }, ...]

// 変換後にバッファをクリア
collector.flush();
```

### 変換ルール

| フィードバック | 分類 | 条件 |
|---|---|---|
| `click` | ✅ positive | 常に |
| `dwell` | ✅ positive | `value >= dwellThresholdMs` |
| `dwell` | ❌ negative | `value < dwellThresholdMs` |
| `skip` | ❌ negative | 常に |
| (未操作) | ❌ negative | positive が存在する場合 |

---

## AdaptiveScheduler

学習率の自動減衰とバッチ学習のタイミング制御を行います。

```typescript
import { TripletTrainer, AdaptiveScheduler } from "@warpvector/ml";

const trainer = new TripletTrainer(1536);
const scheduler = new AdaptiveScheduler(trainer, {
  initialLearningRate: 0.01,   // 初期学習率
  minLearningRate: 0.0001,     // 最小学習率
  decayRate: 0.001,            // 減衰率
  batchSize: 5,                // 5件溜まったら自動学習
  maxBufferSize: 100,          // バッファ上限
});

// フィードバック → 学習データ → スケジューラーに投入
const examples = collector.toTripletExamples();
const updated = await scheduler.addFeedback(currentWeights, examples);

if (updated) {
  // batchSize に達して学習が実行された
  currentWeights = updated;
  console.log(`LR: ${scheduler.currentLearningRate}`);
  console.log(`Total steps: ${scheduler.totalSteps}`);
}
```

### 学習率の減衰

```
lr(n) = max(minLR, initialLR / (1 + decayRate × n))
```

| 学習回数 (n) | 学習率 (デフォルト設定) |
|---|---|
| 0 | 0.0100 |
| 100 | 0.0091 |
| 1,000 | 0.0050 |
| 10,000 | 0.0001 (下限) |

### 状態の永続化

```typescript
// エクスポート（totalSteps とハイパーパラメータを保存）
const schedulerState = scheduler.exportState();
localStorage.setItem("scheduler", schedulerState);

// インポート（学習率の継続性を保つ）
const restored = AdaptiveScheduler.importState(
  trainer,
  localStorage.getItem("scheduler")!,
);
```

---

## FederatedAggregator

複数ユーザーの学習結果を FedAvg で集約し、グローバルベースラインを更新します。

```typescript
import { FederatedAggregator } from "@warpvector/ml";

const aggregator = new FederatedAggregator(globalBaseWeights, 1536);

// 各クライアントの学習済み重みを登録
aggregator.submitUpdate({
  weights: clientAWeights,
  interactionCount: 100,  // 100回学習 → 信頼度高
});

aggregator.submitUpdate({
  weights: clientBWeights,
  interactionCount: 25,   // 25回学習 → 信頼度低
});

// FedAvg で集約
const newGlobalBase = aggregator.aggregate();

// 次のラウンドの準備
aggregator.reset(newGlobalBase);
```

### 集約アルゴリズム (FedAvg)

```
W_new = W_base + Σ (count_i / total_count) × (W_i − W_base)
```

`interactionCount` が多いクライアントほど、集約結果への寄与が大きくなります。

---

## 実践例: E コマース検索の全体フロー

```typescript
import {
  FeedbackCollector,
  AdaptiveScheduler,
  TripletTrainer,
} from "@warpvector/ml";
import { IntentAdapter } from "@warpvector/core";

// 初期化
const dim = 1536;
const collector = new FeedbackCollector();
const trainer = new TripletTrainer(dim);
const scheduler = new AdaptiveScheduler(trainer, { batchSize: 5 });
const adapter = new IntentAdapter(dim);

let weights = loadWeightsFromStorage() ?? adapter.getIdentityWeights();

// --- ユーザーが検索するたび ---

// 1. 検索結果を表示
const impId = collector.recordImpression({
  queryVector: await embed(query),
  resultVectors: results.map(r => r.vector),
  timestamp: Date.now(),
});

// 2. クリックイベント
onResultClick((index) => {
  collector.recordFeedback({ impressionId: impId, resultIndex: index, type: "click" });
});

// 3. 定期的に学習
async function learnFromFeedback() {
  const examples = collector.toTripletExamples();
  if (examples.length === 0) return;

  const updated = await scheduler.addFeedback(weights, examples);
  if (updated) {
    weights = updated;
    adapter.addIntent("search", weights);
    saveWeightsToStorage(weights);
  }
  collector.flush();
}

// 4. 30秒ごとに学習を実行
setInterval(learnFromFeedback, 30_000);
```
