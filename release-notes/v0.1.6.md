# 🌌 v0.1.6 — Zero-Dependency Core & Feedback Loop / ゼロ依存コアへの刷新とフィードバックループ

We have restructured the package to make the core fully zero-dependency and introduced feedback loop capabilities under `@warpvector/ml`.

コア機能を「完全ゼロ依存」化し、さらに `@warpvector/ml` にフィードバックループ機能を導入しました。

---

## 🇬🇧 English Release Notes

### 🚀 New Features (`@warpvector/ml`)
Introduced feedback loop capabilities to support adaptive learning and federated learning on edge environments:
- **`FeedbackCollector`**: Collects user explicit/implicit interactions.
- **`AdaptiveScheduler`**: Manages learning schedules based on feedback.
- **`FederatedAggregator`**: Aggregates weight updates across edge nodes.

### 🏗 Architecture & Bundling Improvements
- **Fully Zero-Dependency Core**: Eliminated external runtime dependencies from the main entrypoint (`warpvector`). You can now use core adapters/trainers without installing Prisma or LangChain.
- **Dedicated Entrypoints**: Cleaned up package separation for integrations:
  - Core: `import { ... } from "warpvector"`
  - Prisma Integration: `import { ... } from "warpvector/prisma"`
  - LangChain Integration: `import { ... } from "warpvector/langchain"`
- **Self-Contained Bundle**: Configured `tsup` to bundle internal `@warpvector/*` workspaces under `warpvector`, resolving dependency resolution errors (`workspace:*` protocol) for external consumers.

### 🐛 Bug Fixes & Developer Experience (DX)
- **TypeScript Resolution (rootDir) Fix**: Set up isolated `tsconfig.json` configurations per sub-package to clear IDE root directory constraint warnings.
- **Refined PeerDependencies**: Configured Prisma and LangChain as optional peer dependencies to improve compatibility.
- **Documentation**: Updated the README to clearly document specific dependency prerequisites for integrations.

---

## 🇯🇵 日本語リリースノート

### 🚀 新機能 (Features)
フィードバックループ機能の追加 (@warpvector/ml): 適応的な学習やフェデレーテッドラーニングをサポートするための機能を追加しました。
- **`FeedbackCollector`**: ユーザーからのフィードバックを収集。
- **`AdaptiveScheduler`**: フィードバックに基づく適応的なスケジューリング。
- **`FederatedAggregator`**: 分散環境での重み集約。

### 🏗 アーキテクチャの改善 (Architecture & Bundling)
- **コア機能の「完全ゼロ依存」化**: `warpvector` のメインエントリポイントから外部依存を排除し、コア機能（各種 Adapter, Trainer 等）を利用する際に、Prisma や LangChain をインストールしなくてもエラーが発生しないようになりました。
- **独立したエントリポイントの提供**: Prisma や LangChain との統合機能を利用する場合は、個別のエントリポイントからインポートする構成に変更しました。
  - コア機能: `import { ... } from "warpvector"`
  - Prisma統合: `import { ... } from "warpvector/prisma"`
  - LangChain統合: `import { ... } from "warpvector/langchain"`
- **サブパッケージの完全バンドル化**: `tsup` の設定を見直し、npm install warpvector 時に内部の `@warpvector/*` パッケージをすべてバンドルに含めることで、消費者が不要なワークスペース依存関係（`workspace:*`）でインストールエラーになる問題を根本的に解決しました。

### 🐛 バグ修正・開発体験 (Bug Fixes & DX)
- **TypeScriptの解決エラー (rootDir) の修正**: IDE向けに各サブパッケージ専用の `tsconfig.json` を整備し、エディタ上でファイルを開いた際に表示されていた `rootDir` 関連の制約エラーを解消しました。
- **peerDependencies の適切な宣言**: Prisma / LangChain 等の関連パッケージを `peerDependencies` (optional: true) に設定し、パッケージマネージャーでの解決が適切に行われるようにしました。
- **README のアップデート**: インストール手順にて、統合機能ごとの追加インストール要件（`@prisma/client` 等）を明記しました。
