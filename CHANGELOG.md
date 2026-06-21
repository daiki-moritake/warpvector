# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-06-21

### Added
- `scripts/sync-versions.ts`: ルートの `package.json` から全サブパッケージへバージョンを自動同期するスクリプト
- `packages/*/tsconfig.build.json`: DTS ビルド専用の tsconfig（`paths` を無効化し `rootDir` エラーを解消）
- `docs/10-projection-migration.md`: ProjectionAdapter と MigrationTrainer のドキュメント
- `docs/11-task-arithmetic.md`: TaskArithmetic のドキュメント
- `docs/12-vsa.md`: VsaAdapter (超次元計算 / Binary VSA) のドキュメント
- README に §11〜§15（ProjectionAdapter, MigrationTrainer, TaskArithmetic, VsaAdapter, Slerp/Reject）を追加

### Fixed
- README のコード例を実際の API シグネチャに修正（`addMlp()` → `addStep()`, `quantize()` → `setFinalStage()`, InfoNCETrainer / MlpAdapter の引数）
- `docs/api-reference.md` を全面更新（存在しないメソッド削除、全12クラス・全型定義を網羅）
- `docs/7-trainers.md` の TripletTrainer / InfoNCETrainer コード例を実際の API に修正

### Changed
- 全サブパッケージのビルドスクリプトを `--tsconfig tsconfig.build.json` に変更
- `build:packages` の先頭で `sync:versions` を自動実行するように変更

## [0.1.3] - 2026-06-20

### Added
- `FinalStageAdapter` インターフェース: パイプライン最終段に量子化等を安全に配置するための型安全な設計
- `WarpPipeline.setFinalStage()`: `FinalStageAdapter` をパイプライン末尾に設定
- `WarpPipeline.registerFinalStage()`: `FinalStageAdapter` のレジストリ登録
- `validation.ts`: `importState` 用の軽量バリデーションユーティリティ群
  - `safeJsonParse`, `assertPositiveInt`, `assertNonNegativeInt`, `assertNumberArray`, `assertObject`, `assertArray`, `assertType`
- `MlpAdapter.setLayerWeights()`: 実行時のレイヤー重み更新メソッド
- CI ワークフロー (`.github/workflows/ci.yml`): PR/push ごとのテスト自動実行
- E2E チュートリアル (`docs/tutorial.md`)
- 精度ベンチマーク (`benchmarks/accuracy.ts`)
- テスト50件追加 (97 → 147)

### Changed
- `WarpPipeline.run()` / `runBatch()`: 中間ベクトルを `Float32Array` に統一、`as InputVector` キャストを排除
- `WarpPipeline.exportState()`: 戻り値が `{ steps, finalStage? }` 形式に変更（旧形式 `PipelineState[]` との後方互換あり）
- `MlpAdapter`: WASM 重みを `init()` 時に永続領域に書き込み、`tune()` では入出力バッファのみスタック管理
- `ProjectionAdapter.tune()`: WASM メモリ管理を `withWasmMemoryStack` + `allocateWasmMemory` パターンに統一
- 全アダプタの `importState`: バリデーション付きデシリアライゼーションに強化

### Removed
- `IntentAdapter.tuneBatch()` / `tuneBatchBlended()` の未使用 `requiredBytes` 変数
