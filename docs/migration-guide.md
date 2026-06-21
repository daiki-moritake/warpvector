# マイグレーションガイド (v0.1 → v0.2)

WarpVector v0.2 では、スケーラビリティと開発者体験 (DX) を大幅に強化しました。
既存のコードとの後方互換性は維持されていますが、新機能を活用するための移行手順を以下にまとめます。

---

## 破壊的変更

> **v0.2 には破壊的変更はありません。** 既存のコードはそのまま動作します。

ただし、以下の変更により既存のエラーハンドリングコードが影響を受ける可能性があります。

### バリデーションエラーの型が変更

バリデーション関数（`assertType`, `assertArray` 等）がスローするエラーが、`Error` から `WarpValidationError` に変更されました。

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

> `WarpValidationError` は `Error` を継承しているため、`catch (e: Error)` で従来通りキャッチできます。

---

## 新機能の活用

### 1. パイプラインの自動初期化 (`autoInit`)

**v0.1 以前:**
```typescript
const pipeline = new WarpPipeline(1536).addStep("mlp", mlpAdapter);
await pipeline.init(); // 忘れると動作不正
pipeline.run(vector);
```

**v0.2 以降:**
```typescript
const pipeline = new WarpPipeline(1536).addStep("mlp", mlpAdapter);
// init() を呼ばなくてOK — runStream() の初回で自動初期化
for await (const result of pipeline.runStream(vectors)) { /* ... */ }
```

> `autoInit` はデフォルトで有効です。無効にしたい場合は `new WarpPipeline(1536, { autoInit: false })`。

### 2. 構造化エラーによるデバッグ改善

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

### 3. メトリクス収集

```typescript
pipeline.metrics.enable();
pipeline.run(vector);

const stats = pipeline.metrics.getMetrics();
console.log(`Avg: ${stats.avgRunDurationMs.toFixed(2)}ms`);
console.log("Per step:", stats.avgStepDurationMs);
```

### 4. デバッグ支援 (`inspect` / `dryRun`)

```typescript
// パイプライン構成の確認
console.log(pipeline.inspect());

// 各ステップの中間出力を確認
const debug = pipeline.dryRun(testVector, { intent: "tech" });
debug.forEach(r => console.log(`${r.step}: dim=${r.output.length}`));
```

### 5. WASM メモリ監視

```typescript
import { getWasmMemoryStats } from 'warpvector';

const stats = getWasmMemoryStats();
console.log(`Peak memory: ${(stats.peakBytes / 1024).toFixed(0)}KB`);
```

---

## 新しいエラークラス一覧

| クラス | 用途 |
|--------|------|
| `WarpError` | 全 WarpVector エラーの基底クラス |
| `WarpPipelineError` | パイプラインの特定ステップでの失敗（`stepIndex`, `stepType` 付き） |
| `WarpDimensionMismatchError` | 次元不一致（`expectedDim`, `actualDim` 付き） |
| `WarpInitializationError` | 初期化未完了でのメソッド呼び出し |
| `WarpValidationError` | importState/設定のバリデーション失敗（`component`, `field` 付き） |

---

## 推奨される移行手順

1. **依存を更新**: `npm install warpvector@latest`
2. **`init()` の呼び出しをレビュー**: `autoInit` により不要な場合は削除可能
3. **エラーハンドリングを強化**: `WarpPipelineError` を活用して、どのステップで失敗したか特定可能に
4. **メトリクスを有効化**: 開発環境で `pipeline.metrics.enable()` を追加してボトルネックを特定
5. **`inspect()` を活用**: パイプライン構成の確認にデバッグ出力を利用
