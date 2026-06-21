# トラブルシューティングガイド

WarpVector でよくある問題と解決策をまとめています。

---

## 1. WASM 初期化エラー

### 症状
```
WASM initialization failed, falling back to JS.
```

### 原因
- WebAssembly がサポートされていない環境で実行している
- ランタイムの WASM 実行ポリシーで制限されている（一部の Cloudflare Workers プランなど）

### 解決策
1. **自動フォールバック**: WarpVector は WASM 初期化に失敗した場合、自動的に純粋な JS 実装にフォールバックします。この警告メッセージは無害ですが、パフォーマンスは低下します。
2. **明示的な初期化**: `await pipeline.init()` を起動時に呼ぶことで、WASM の初期化失敗を早期に検出できます。
3. **autoInit の使用**: v0.2.0 以降、`WarpPipeline` はデフォルトで `autoInit: true` です。初回の `runStream()` 呼び出し時に自動で WASM を初期化します。

---

## 2. init() の呼び忘れ

### 症状
- `MlpAdapter` の結果がゼロベクトルになる
- WASM関連のエラーがスローされる

### 解決策

**v0.2.0 以降（推奨）**: `WarpPipeline` の `autoInit` 機能を使用する（デフォルト有効）:

```typescript
const pipeline = new WarpPipeline(1536); // autoInit はデフォルトで true
// init() を呼ばなくても、初回の runStream() で自動初期化される
```

**明示的に初期化する場合**:
```typescript
const pipeline = new WarpPipeline(1536, { autoInit: false });
await pipeline.init(); // 明示的に呼ぶ
```

---

## 3. 次元不一致エラー

### 症状
```
IntentAdapter: 入力ベクトルの次元が一致しません。
  期待: 1536
  実際: 768
```

### 原因
- 埋め込みモデルを変更したが、アダプターの次元設定を更新していない
- `ProjectionAdapter` で次元変換した後のパイプラインステップが、変換前の次元を期待している

### 解決策
1. **`pipeline.inspect()` で構成を確認**:
```typescript
console.log(pipeline.inspect());
// Pipeline [1536-dim]
//   Step 0: MlpAdapter
//   Step 1: IntentAdapter
//   Final: QuantizationAdapter
```

2. **`pipeline.dryRun()` で中間出力を確認**:
```typescript
const results = pipeline.dryRun(testVector, { intent: "tech" });
results.forEach(r => {
  console.log(`${r.step}: dim=${r.output.length}, ${r.durationMs.toFixed(2)}ms`);
});
```

3. **モデル移行時**: `MigrationTrainer` を使って既存のベクトルを新モデルの空間に変換する射影行列を学習してください。

---

## 4. エッジ環境でのメモリ制限

### 症状
- `WASM memory grow failed` のエラー
- OOM (Out Of Memory) で Worker がクラッシュする

### 原因
- Cloudflare Workers のメモリ制限（128MB）に到達した
- 大量のベクトルを一括で処理しようとしている

### 解決策

1. **ストリーム処理を使用する**:
```typescript
// 一括処理ではなくストリームで処理（メモリ効率的）
const results = pipeline.runStream(vectorGenerator, {
  batchSize: 64,  // バッチサイズを小さくする
});
```

2. **量子化を使用してメモリを削減**:
```typescript
// Float32 → Int8 で 1/4、Binary で 1/32 に圧縮
pipeline.setFinalStage("quantize", new QuantizationAdapter({
  type: "int8",  // または "binary"
  dim: 1536
}));
```

3. **WASMメモリ使用量の監視**:
```typescript
import { getWasmMemoryStats } from 'warpvector';

const stats = getWasmMemoryStats();
console.log(`Used: ${stats.usedBytes}, Peak: ${stats.peakBytes}, Total: ${stats.totalBytes}`);
```

---

## 5. パフォーマンスのボトルネック特定

### 症状
- パイプラインの処理速度が期待より遅い
- どのステップがボトルネックか分からない

### 解決策

**メトリクス収集を有効にする**:
```typescript
const pipeline = new WarpPipeline(1536)
  .addStep("mlp", mlpAdapter)
  .addIntent(intents);

// メトリクス収集を有効化
pipeline.metrics.enable();

// 処理を実行
for (const vec of vectors) {
  pipeline.run(vec, { intent: "tech" });
}

// 結果を確認
const metrics = pipeline.metrics.getMetrics();
console.log(`Total runs: ${metrics.totalRuns}`);
console.log(`Avg run duration: ${metrics.avgRunDurationMs.toFixed(2)}ms`);
console.log("Step durations:");
for (const [step, avg] of Object.entries(metrics.avgStepDurationMs)) {
  console.log(`  ${step}: ${avg.toFixed(3)}ms`);
}
```

**`dryRun()` で単一ベクトルを計測する**:
```typescript
const results = pipeline.dryRun(testVector, { intent: "tech" });
results.forEach(r => {
  console.log(`${r.step}: ${r.durationMs.toFixed(3)}ms`);
});
```

---

## 6. 並行処理でのデータ破壊

### 症状
- 複数リクエストを同時処理するとベクトルの結果がおかしくなる

### 原因
- WASM メモリは共有リソースのため、複数の処理が同時にアクセスすると破壊が起きる

### 解決策
**`runStream` を使用する**: v0.2.0 以降、`runStream` は内部で `wasmMutex` による排他制御を行います。

**手動で排他制御する場合**:
```typescript
import { wasmMutex } from 'warpvector';

// WASM メモリを使う処理を排他的に実行
const result = await wasmMutex.runExclusive(() => {
  return pipeline.run(vector, { intent: "tech" });
});
```

---

## 7. importState でのエラー

### 症状
```
WarpVector: フィールド 'state' のバリデーションに失敗しました。
  JSON文字列が必要ですが、object 型が渡されました。
```

### 原因
- `importState()` に `exportState()` の結果を直接渡さず、加工してしまった
- JSON のシリアライズ/デシリアライズの過程でデータが壊れた

### 解決策
```typescript
// 正しい使い方
const state = adapter.exportState();    // string 型
const json = JSON.stringify(state);     // Redis 等に保存する場合

// 復元時
const parsed = JSON.parse(json);        // string に戻す
const restored = IntentAdapter.importState(parsed);
```
