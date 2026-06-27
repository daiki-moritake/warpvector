# マイグレーションガイド

各メジャーバージョンアップグレードの破壊的変更とマイグレーション手順をまとめています。

- [v0.3 → v0.4](#v03--v04)（最新）
- [v0.2 → v0.3](#v02--v03)
- [v0.1 → v0.2](#v01--v02)

---

## v0.3 → v0.4

### 破壊的変更

#### 1. QuantizationAdapter API の変更

`tune()` メソッドが廃止されました。すべての量子化操作に `encode()` を使用してください。

```diff
- const quantized = quantizer.tune(vector);
+ const quantized = quantizer.encode(vector);
```

#### 2. トレーニングユーティリティの `@warpvector/train` への移動

`SoftWhiteningAdapter` および関連するトレーニングモジュールが `@warpvector/ml` から新しい `@warpvector/train` パッケージに移動しました。

```diff
- import { SoftWhiteningAdapter } from 'warpvector/ml';
+ import { SoftWhiteningAdapter } from 'warpvector/train';
```

```bash
# サブパッケージを直接使用している場合:
npm install @warpvector/train
```

#### 3. Adam Optimizer の ML パッケージからの削除

組み込みの Adam optimizer が `@warpvector/ml` から削除されました。トレーニング関連のタスクには `@warpvector/train` を使用してください。

### v0.4 の新パッケージ

| パッケージ | 用途 |
|-----------|------|
| `@warpvector/train` | トレーニング、ファインチューニング、Auto-ML ツール |
| `@warpvector/rerank` | ColBERT および高度なグラフベースリランカー |
| `@warpvector/eval` | RAG 評価キット（Precision@K, Recall@K, NDCG, MRR, MAP） |

### マイグレーション手順

1. **依存関係を更新**: `npm install warpvector@latest`
2. **`tune()` 呼び出しを置換**: `quantizer.tune(v)` → `quantizer.encode(v)` に変更
3. **トレーニング関連のインポートを更新**: `SoftWhiteningAdapter` のインポート元を `warpvector/ml` から `warpvector/train` に変更
4. **新パッケージのインストール**（サブパッケージ利用の場合）: `npm install @warpvector/train @warpvector/rerank @warpvector/eval`

---

## v0.2 → v0.3

### 破壊的変更

> **v0.3 には破壊的変更はありません。** 既存の v0.2 コードはそのまま動作します。

### 新機能

- **WarpTracer**: ゼロ依存の OpenTelemetry 互換トレーシング
- **Cloudflare Vectorize** との統合（`VectorDBAdapter`）
- **IntentMatrixFactory**: カテゴリサンプルからインテント行列を自動学習
- **`@warpvector/experimental`** パッケージ（不安定な機能向け）

### マイグレーション手順

1. **依存関係を更新**: `npm install warpvector@latest`
2. **オプション**: 本番環境の監視に `WarpTracer` を追加
3. **オプション**: 手作りのインテント行列の代わりに `IntentMatrixFactory` を使用

---

## v0.1 → v0.2

### 破壊的変更

> **v0.2 には破壊的変更はありません。** 既存のコードはそのまま動作します。

ただし、既存のエラーハンドリングコードは以下の変更の影響を受ける可能性があります：

#### バリデーションエラー型の変更

バリデーション関数（`assertType`、`assertArray` など）がスローするエラーが `Error` から `WarpValidationError` に変更されました。

```diff
- catch (e) {
-   // e.message: "Invalid state: field 'matrix' must be an array"
- }
+ catch (e) {
+   if (e instanceof WarpValidationError) {
+     console.error(`${e.component}: ${e.field} - ${e.message}`);
+   }
+ }
```

> `WarpValidationError` は `Error` を継承しているため、従来通り `catch (e: Error)` でキャッチできます。

---

### 新機能の活用

#### 1. パイプラインの自動初期化 (`autoInit`)

**v0.1 以前:**
```typescript
const pipeline = new WarpPipeline(1536).addStep("mlp", mlpAdapter);
await pipeline.init(); // 忘れるとバグの原因に
pipeline.run(vector);
```

**v0.2 以降:**
```typescript
const pipeline = new WarpPipeline(1536).addStep("mlp", mlpAdapter);
// init() は省略可能 — runStream() 初回呼び出し時に自動初期化
for await (const result of pipeline.runStream(vectors)) { /* ... */ }
```

> `autoInit` はデフォルトで有効です。無効にする場合は `new WarpPipeline(1536, { autoInit: false })` を使用してください。

#### 2. 構造化エラーによるデバッグ改善

**v0.1 以前:**
```typescript
try {
  pipeline.run(vector);
} catch (e) {
  console.error(e.message); // 汎用的なエラーメッセージ
}
```

**v0.2 以降:**
```typescript
import { WarpPipelineError, WarpValidationError } from 'warpvector';

try {
  pipeline.run(vector);
} catch (e) {
  if (e instanceof WarpPipelineError) {
    console.error(`Step ${e.stepIndex} (${e.stepType}): ${e.message}`);
    console.error("Original cause:", e.cause);
  }
}
```

#### 3. メトリクス収集

```typescript
pipeline.metrics.enable();
pipeline.run(vector);

const stats = pipeline.metrics.getMetrics();
console.log(`平均: ${stats.avgRunDurationMs.toFixed(2)}ms`);
console.log("ステップ別:", stats.avgStepDurationMs);
```

#### 4. デバッグ支援 (`inspect` / `dryRun`)

```typescript
// パイプライン設定の確認
console.log(pipeline.inspect());

// 各ステップの中間出力を確認
const debug = pipeline.dryRun(testVector, { intent: "tech" });
debug.forEach(r => console.log(`${r.step}: dim=${r.output.length}`));
```

#### 5. WASM メモリ監視

```typescript
import { getWasmMemoryStats } from 'warpvector';

const stats = getWasmMemoryStats();
console.log(`ピークメモリ: ${(stats.peakBytes / 1024).toFixed(0)}KB`);
```

---

## エラークラス一覧

| クラス | 用途 |
|--------|------|
| `WarpError` | すべての WarpVector エラーの基底クラス |
| `WarpPipelineError` | 特定のパイプラインステップでの失敗（`stepIndex`、`stepType` を含む） |
| `WarpDimensionMismatchError` | 次元の不一致（`expectedDim`、`actualDim` を含む） |
| `WarpInitializationError` | 初期化完了前のメソッド呼び出し |
| `WarpValidationError` | importState/設定のバリデーション失敗（`component`、`field` を含む） |
