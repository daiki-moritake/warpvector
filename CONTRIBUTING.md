# Contributing to WarpVector

WarpVector への貢献を歓迎します！

## 開発環境のセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/daiki-moritake/warpvector.git
cd warpvector

# 依存関係のインストール（Bun を使用）
bun install

# テストの実行
bun test

# ビルド
bun run build
```

## 開発ワークフロー

### ブランチ戦略

- `main` — リリース可能な安定版
- `feature/*` — 新機能の開発
- `fix/*` — バグ修正

### コミットメッセージ

日本語で記述してください。例：

```
feat: FinalStageAdapterインターフェースを追加
fix: MlpAdapterのWASMメモリリークを修正
test: importStateバリデーションのテストを追加
docs: E2Eチュートリアルを追加
```

### Pull Request

1. `feature/*` または `fix/*` ブランチを作成
2. 変更を実装
3. `bun test` が全てパスすることを確認
4. `bun run build` が成功することを確認
5. PR を作成（CI が自動で実行されます）

## プロジェクト構造

```
warpvector/
├── packages/
│   ├── core/        # コアエンジン（IntentAdapter, ProjectionAdapter, WarpPipeline）
│   ├── ml/          # ML系アダプタ（MlpAdapter, WhiteningAdapter）
│   ├── extras/      # 追加アダプタ（QuantizationAdapter, ColbertAdapter）
│   ├── prisma/      # Prisma連携
│   └── langchain/   # LangChain連携
├── assembly/        # AssemblyScript（WASM）ソース
├── benchmarks/      # パフォーマンスベンチマーク
├── docs/            # ドキュメント
└── examples/        # 使用例
```

## テストの書き方

- テストファイルは `packages/*/tests/` に配置
- `bun:test` を使用
- 境界値テスト（ゼロベクトル、NaN、次元不一致）を必ず含める
- `importState` を変更した場合はバリデーションテストを追加

## WASM 開発

WASM コードは `assembly/index.ts` (AssemblyScript) で記述します。

```bash
# WASM のビルド
bun run build:wasm

# ビルド後、packages/core/src/wasm/wasm-binary.ts が自動生成されます
```

## ライセンス

MIT License - 詳細は [LICENSE](./LICENSE) を参照してください。
