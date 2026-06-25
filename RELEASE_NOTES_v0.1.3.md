# 🌌 v0.1.3 — Reliability & Performance Update / 信頼性と性能の向上

We have released v0.1.3 containing reliability updates, type safety improvements, and performance enhancements across WASM components.

信頼性とパフォーマンス、および型安全性の向上を目的とした大幅なアップデートを行いました。

---

## 🇬🇧 English Release Notes

### Added
- **`FinalStageAdapter` Interface & Methods**: Added a type-safe interface and pipeline API (`WarpPipeline.setFinalStage()` / `registerFinalStage()`) to safely handle late-stage pipeline steps (e.g., quantization).
- **`validation.ts`**: Introduced lightweight validation helpers for state de-serialization (`safeJsonParse`, `assertPositiveInt`, `assertObject`, etc.).
- **`MlpAdapter.setLayerWeights()`**: Dynamic runtime layer weights updating method.
- **CI Workflow**: Configured automated test runs on PR/pushes.
- **E2E Tutorial**: Added a comprehensive tutorial (`docs/tutorial.md`).
- **Benchmark**: Added accuracy benchmarks (`benchmarks/accuracy.ts`).
- **Tests**: Expanded test coverage (+50 unit tests).

### Changed
- **WarpPipeline Vector Formats**: Unified intermediate vector formats to `Float32Array`, eliminating legacy `as InputVector` casting.
- **`WarpPipeline.exportState()`**: Refactored the export schema to `{ steps, finalStage? }` while maintaining backward compatibility.
- **`MlpAdapter`**: Weights are now written to persistent WASM memory once during `init()`, optimizing memory stack handling during `tune()`.
- **`ProjectionAdapter.tune()`**: Coded WASM stack allocations to use the standard `withWasmMemoryStack` pattern.
- **Validation**: Upgraded adapter deserialization (`importState`) with validation asserts.

---

## 🇯🇵 日本語リリースノート

### Added
- **`FinalStageAdapter` インターフェース**: パイプライン最終段に量子化等を安全に配置するための型安全な設計。
- **`WarpPipeline.setFinalStage()` / `registerFinalStage()`**: `FinalStageAdapter` をパイプライン末尾に設定・登録。
- **`validation.ts`**: `importState` 用 of 軽量バリデーションユーティリティ群。
  - `safeJsonParse`, `assertPositiveInt`, `assertNonNegativeInt`, `assertNumberArray`, `assertObject`, `assertArray`, `assertType`
- **`MlpAdapter.setLayerWeights()`**: 実行時のレイヤー重み更新メソッド。
- CI ワークフロー (`.github/workflows/ci.yml`): PR/push ごとのテスト自動実行。
- E2E チュートリアル (`docs/tutorial.md`)。
- 精度ベンチマーク (`benchmarks/accuracy.ts`)。
- テスト50件追加 (97 → 147)。

### Changed
- **`WarpPipeline.run()` / `runBatch()`**: 中間ベクトルを `Float32Array` に統一、`as InputVector` キャストを排除。
- **`WarpPipeline.exportState()`**: 戻り値が `{ steps, finalStage? }` 形式に変更（旧形式 `PipelineState[]` との後方互換あり）。
- **`MlpAdapter`**: WASM 重みを `init()` 時に永続領域に書き込み、`tune()` では入出力バッファのみスタック管理。
- **`ProjectionAdapter.tune()`**: WASM メモリ管理を `withWasmMemoryStack` + `allocateWasmMemory` パターンに統一。
- 全アダプタの `importState`: バリデーション付きデシリアライゼーションに強化。

### Removed
- `IntentAdapter.tuneBatch()` / `tuneBatchBlended()` の未使用 `requiredBytes` 変数を削除。
