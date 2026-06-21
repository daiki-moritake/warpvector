# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-22

### Added

#### 新機能 — ML / アダプタ
- `SoftWhiteningAdapter` (旧 `InverseDiffusionAdapter`): オンライン次元チューニングのための逆拡散ベースの軟白色化アダプタ
- `TimeReversalReranker`: 波動方程式に着想を得たグラフベースのリランキングアダプタ
- `MultipathScatteringReranker`: ランダムウォークによるハブ検出リランカー（WASM 対応）
- `BaseGraphReranker`: グラフベースリランカーの共通基底クラス
- `MoEAdapter`: Mixture of Experts (MoE) ルーティングアダプタ
- `AutoTuningPipeline`: AutoML による自動パイプライン最適化
- `CrossEncoderTrainer`: 高精度再学習のためのクロスエンコーダートレーナー
- Advanced Safety & Safe Quantization: 大次元 MLP 推論のメモリ安全性強化とバリデーション

#### 新機能 — ツール / DX
- `create-warpvector-app` CLI: プロジェクトスキャフォールディングツール
- Interactive Playground: WASM アクセラレーテッドなベクトル空間のインタラクティブデモ
  - PCA による動的 2D 射影カメラ
  - LLM による埋め込み生成とインテント/クエリの動的入力
  - ブレンドモード・ライブコードスニペットパネル
  - 量子化・白色化コントロール
  - コスト削減シミュレーションビジュアル
- GitHub Pages デプロイワークフロー (`deploy-playground.yml`)

#### 新機能 — ドキュメント
- Edge Quickstart ガイド (`docs/edge-quickstart.md`)
- Auto-Learning ガイド (`docs/auto-learning-guide.md`)
- 日本語版ドキュメント全面翻訳（API Reference, Tutorial, Cookbook 等）
- Cookbook: E-commerce Search, Pinecone RAG, Cloudflare Edge
- Migration Guide (`docs/migration-guide.md`): v0.1 → v0.2 アップグレードガイド
- 技術記事 4 本追加（`articles/` ディレクトリ）
- Express Batch API / Edge API Handler のサンプル追加

#### 新機能 — CI / インフラ
- CI ワークフロー拡張: Node.js 20/22 互換性マトリクス、バンドルサイズ監視
- プロジェクトガバナンス: CODE_OF_CONDUCT, CONTRIBUTING, Issue Templates

### Changed
- `InverseDiffusionAdapter` → `SoftWhiteningAdapter` にリネーム
- CI の Node.js バージョンを v22 に更新
- バリデーションエラーメッセージのローカライズ
- WASM バイナリストレージを固定長バイト配列に変更し、リランカーロジックを最適化
- Trainer インターフェースの改善: トレーニングループとカスタムトレーニングパイプライン対応
- Pipeline アダプタインターフェースのバッチ処理サポート強化
- Playground エンジンのステート遷移ロジック簡素化

### テスト
- テスト数: 147 → 209 (62 テスト追加)
- バンドルサイズ: index.mjs 112KB (24.0KB gzipped)

---

## [0.1.7] - 2026-06-21

### Added
- `FeedbackCollector`, `AdaptiveScheduler`, `FederatedAggregator`: 検索システムにおけるオンライン学習および連合学習（Federated Learning）機能を追加

### Changed
- `FederatedAggregator`: 内部での配列次元バリデーションおよびフラット化処理を `@warpvector/core` の `getFlatMatrixAndBias` に共通化
- `AdaptiveScheduler`: `addFeedback` にて、バッファに溜まったデータを一度の呼び出しで全てバッチ処理（ループ処理）するように修正

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
