# Intent Matrix Factory — ゼロ設定でIntent行列を自動生成

Intent Warping は WarpVector の最も強力な機能ですが、これまでは **Intent行列（変換行列 W とバイアス b）をユーザーが自分で用意する必要がありました**。

`IntentMatrixFactory` は、この最大のペインポイントを解決します。**カテゴリごとに5〜10個のサンプルベクトルを追加するだけで、最適なIntent行列が自動的に学習・生成されます。**

## クイックスタート

```typescript
import { IntentMatrixFactory } from 'warpvector/ml';
import { IntentAdapter, WarpPipeline } from 'warpvector';

// 1. ファクトリを作成（次元数を指定）
const factory = new IntentMatrixFactory(1536);

// 2. カテゴリごとにサンプルベクトルを追加
//    ※ embed() はお使いの Embedding API（OpenAI, Cohere等）の関数
factory.addCategory("tech", [
  await embed("TypeScript のランタイム性能最適化"),
  await embed("WebAssembly SIMD ベクトル化技術"),
  await embed("エッジコンピューティングのレイテンシ削減"),
  await embed("Bun vs Node.js ベンチマーク比較"),
  await embed("ゼロコピーメモリバッファ管理"),
]);

factory.addCategory("business", [
  await embed("Q4 収益予測と成長見通し"),
  await embed("顧客獲得コストの最適化"),
  await embed("SaaS業界の市場シェア分析"),
  await embed("競合分析とポジショニング戦略"),
  await embed("年間経常収益のトレンド"),
]);

// 3. 自動で最適な Intent 行列を学習（数百ミリ秒〜数秒）
const intents = await factory.build();
// → { tech: { matrix, bias, routingVector }, business: { matrix, bias, routingVector } }

// 4. そのまま IntentAdapter / WarpPipeline に投入
const pipeline = new WarpPipeline(1536).addIntent(intents);

// 5. 検索時に Intent を指定して空間を歪める
const techQuery = pipeline.run(queryVector, { intent: "tech" });
const bizQuery = pipeline.run(queryVector, { intent: "business" });

// 6. あるいは Auto-blending で自動選択
const autoQuery = pipeline.run(queryVector, { autoBlend: true });
```

## 仕組み

`IntentMatrixFactory` は内部で **InfoNCE（対照学習）** を用いて Intent 行列を学習します。

1. **対照学習データの自動構築**: 各カテゴリの anchor-positive-negatives ペアを自動生成
2. **Adam オプティマイザで最適化**: カテゴリ間の分離を最大化するアフィン変換（W, b）を学習
3. **Routing Vector の自動設定**: カテゴリの平均ベクトルを正規化して `tuneAutoBlended()` に対応

```
カテゴリA のベクトル群 → ┐
                        ├→ InfoNCE Loss → Adam Optimizer → Intent行列 (W, b)
カテゴリB のベクトル群 → ┘
```

## API リファレンス

### コンストラクタ

```typescript
const factory = new IntentMatrixFactory(dimension: number);
```

| パラメータ | 型 | 説明 |
|-----------|------|------|
| `dimension` | `number` | 入力ベクトルの次元数（例: 1536） |

### `addCategory()`

カテゴリにサンプルベクトルを追加します。メソッドチェーン対応。

```typescript
factory
  .addCategory("tech", [techVec1, techVec2, techVec3])
  .addCategory("business", [bizVec1, bizVec2, bizVec3]);
```

| パラメータ | 型 | 説明 |
|-----------|------|------|
| `categoryName` | `string` | カテゴリ名（= Intent名として使用） |
| `vectors` | `(number[] \| Float32Array)[]` | サンプルベクトルの配列 |

> **推奨サンプル数**: カテゴリあたり 5〜10 個。少なすぎると学習が不安定になり、多すぎても大きな改善は見られません。

### `build()`

登録されたカテゴリから Intent 行列を自動生成します。

```typescript
const intents = await factory.build(options?: IntentMatrixFactoryOptions);
```

#### `IntentMatrixFactoryOptions`

| オプション | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `training` | `BaseTrainingOptions` | `{ epochs: 200, learningRate: 0.01, autoTune: true, patience: 15 }` | 学習ハイパーパラメータ |
| `temperature` | `number` | `0.07` | InfoNCE の温度パラメータ。小さいほど正解/不正解の分離が鋭くなる |
| `generateRoutingVectors` | `boolean` | `true` | `routingVector` を自動生成するか |
| `maxNegativesPerAnchor` | `number` | `7` | 各 anchor に対する negative サンプルの最大数 |

### ユーティリティメソッド

```typescript
factory.getCategoryNames();      // → ["tech", "business"]
factory.getSampleCount("tech");  // → 5
factory.getTotalSampleCount();   // → 10
```

## ベストプラクティス

### 1. カテゴリはユーザーの「検索意図」に対応させる

✅ 良い例:
```
"product_search"   → 商品を探すクエリのサンプル
"support_faq"      → サポート質問のサンプル
"competitor_analysis" → 競合分析のサンプル
```

❌ 悪い例:
```
"short_query"   → クエリの長さは Intent ではない
"english"       → 言語は Intent ではない
```

### 2. サンプルの多様性を確保する

同じカテゴリ内でも、異なる表現のサンプルを使用してください：

```typescript
// ✅ 良い: 多様な表現
factory.addCategory("tech", [
  await embed("TypeScript のパフォーマンス"),    // 一般的
  await embed("WASM SIMD を最適化する方法"),     // How-to 形式
  await embed("Node.js vs Bun ベンチマーク"),     // 比較形式
  await embed("WebSocket レイテンシプロファイリング"), // 具体的ツール
]);

// ❌ 悪い: ほぼ同じ表現の繰り返し
factory.addCategory("tech", [
  await embed("TypeScript のパフォーマンス"),
  await embed("TypeScript の速度"),
  await embed("TypeScript の最適化"),
]);
```

### 3. オンライン学習との組み合わせ

`IntentMatrixFactory` で初期行列を生成し、`IntentTrainer.updateOnline()` でフィードバックから継続学習するのが最も強力なパターンです：

```typescript
// 初期行列の生成
const intents = await factory.build();

// 運用開始後、ユーザーフィードバックから微調整
const trainer = new IntentTrainer(1536);
const updatedWeights = await trainer.updateOnline(
  intents["tech"],
  { input: queryVector, target: clickedDocVector },
);
```

## 次のステップ

- [Trainers（学習エンジン）](./7-trainers.ja.md) — オンライン学習エンジンの詳細
- [自動学習ガイド](./auto-learning-guide.ja.md) — フィードバックループの実装ガイド
- [インテグレーション](./8-integrations.ja.md) — LangChain / LlamaIndex との統合
