# 自動学習（Auto-Learning）実装ガイド

Warpvectorの最大の特徴は、ユーザーの検索・クリック行動から「自動で空間を最適化する学習ループ」を組み込める点です。
このドキュメントでは、外部のPythonサーバーやGPUを使うことなく、**TypeScript環境（Node.js, Edgeなど）だけで完結するエンドツーエンドの自動学習パイプライン**の実装方法を解説します。

## 全体アーキテクチャ

自動学習は以下のコンポーネントの連携で実現されます。

1. **`FeedbackCollector`**: ユーザーのクリックや滞在時間のログを受け取り、学習可能なデータセット（InfoNCEやTriplet形式）に変換します。
2. **`AdaptiveScheduler`**: 一定件数のデータが溜まったタイミングで自動的に学習を発火させ、同時に学習率（Learning Rate）を最適に減衰させます。
3. **`InfoNCETrainer` / `TripletTrainer`**: Adamオプティマイザーを内蔵したWASM駆動の学習エンジンです。ベクトルの空間を歪める重み（Weights）を更新します。
4. **ストレージ**: 更新された重みをRedisやCloudflare KVなどのKVSに保存し、次回の検索に反映します。

## 実装ステップ

### 1. 初期化

アプリケーションの起動時、またはエッジ環境のグローバルスコープで学習に必要なインスタンスを初期化します。

```typescript
import { 
  FeedbackCollector, 
  AdaptiveScheduler, 
  InfoNCETrainer 
} from 'warpvector/ml';

// 1. トレーナーの初期化（1536次元ベクトルを想定）
// 複数Negativeに対応し、高い精度が出るInfoNCETrainerを推奨します。
const trainer = new InfoNCETrainer(1536);

// 2. スケジューラーの設定
// batchSize: 何件のログが溜まったら学習を発火させるか
// initialLearningRate: 初期の学習率
const scheduler = new AdaptiveScheduler(trainer, { 
  batchSize: 10,
  initialLearningRate: 0.01 
});

// 3. コレクターの設定
// dwellThresholdMs: この時間（ミリ秒）以上滞在したら「正解（ポジティブ）」とみなす
const collector = new FeedbackCollector({ dwellThresholdMs: 3000 });
```

### 2. ログの収集とフィードバックの記録

ユーザーが検索を行い、結果に対してアクション（クリックなど）を起こした際に、コレクターにログを記録します。

```typescript
// 検索時のリクエスト情報を記録し、impressionId（インプレッションID）を発行します。
const impressionId = collector.recordImpression({
  queryVector: rawQueryVector,            // 検索時のクエリベクトル
  resultVectors: [docVec1, docVec2, docVec3], // 表示された検索結果のベクトル一覧
  timestamp: Date.now()
});

// ユーザーが「1番目の結果」をクリックした場合の記録
collector.recordFeedback({
  impressionId: impressionId,
  resultIndex: 0,   // クリックされた結果のインデックス
  type: "click"     // または "dwell", "skip"
});

// （別の検索例）ユーザーが「3番目の結果」を5秒間閲覧（dwell）した場合の記録
collector.recordFeedback({
  impressionId: anotherImpId,
  resultIndex: 2,
  type: "dwell",
  value: 5000       // 滞在時間（ミリ秒）
});
```

### 3. 自動学習の発火と重み更新

ユーザーアクションを受け取るAPIエンドポイント等で、学習データへの変換とスケジューラーへの投入を行います。

```typescript
async function handleUserAction() {
  // 1. コレクターに溜まったログをInfoNCE学習データ形式に変換
  const examples = collector.toInfoNCEExamples();
  if (examples.length === 0) return;

  // 2. 現在の「空間を歪める重み」をDBやKVから取得（初回は初期値）
  const currentWeights = await kvStore.get("intent_weights") || adapter.getIdentityWeights();

  // 3. スケジューラーにデータを投入
  // ※ 内部でログ件数が batchSize (10件) に達しているか判定され、
  //    達していれば WASM 上で Adam オプティマイザーによる学習が走ります。
  const updatedWeights = await scheduler.addFeedback(currentWeights, examples);

  // 4. 学習が発火して重みが更新された場合のみ、ストレージに保存
  if (updatedWeights) {
    console.log("🔥 自動学習が実行され、重みが更新されました！", scheduler.currentLearningRate);
    await kvStore.put("intent_weights", updatedWeights);
    
    // コレクターのバッファをクリアして次のバッチに備える
    collector.flush();
  }
}
```

### 4. 検索（推論）への適用

更新された重みは、次に検索リクエストが来た際に `IntentAdapter` を通じて反映されます。これだけで、ユーザーがクリックしやすい方向にベクトル空間が動的に変化します。

```typescript
import { IntentAdapter } from 'warpvector';

async function performSearch(queryVector) {
  // 1. ストレージから最新の学習済み重みを取得
  const latestWeights = await kvStore.get("intent_weights");
  
  // 2. アダプターに適用
  const adapter = new IntentAdapter(1536);
  adapter.addIntent("auto_learned", latestWeights);
  
  // 3. クエリベクトルをワープ（歪める）
  const optimizedQuery = adapter.tune(queryVector, "auto_learned");
  
  // 4. Vector DB への検索を実行
  return await pinecone.query({ vector: optimizedQuery, topK: 10 });
}
```

## 次のステップ

- 複数のエッジ（クライアント）で学習した結果を中央で統合する分散学習アーキテクチャについては、[Federated Learning（連合学習）](./13-feedback-loop.md) をご参照ください。
- ベースとなるTrainerの数学的背景や種類については [Trainers](./7-trainers.md) をご参照ください。
