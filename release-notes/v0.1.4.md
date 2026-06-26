# 🌌 v0.1.4 — Docs Refresh & Build Pipeline Improvement / ドキュメント刷新とビルド改善

We have fully synchronized and corrected the documentation with our codebase implementation and improved our package build pipeline.

ドキュメントと実装の整合性を全面的に修正し、ビルドパイプラインを改善しました。

---

## 🇬🇧 English Release Notes

### ✨ Highlights
- **Documentation Overhaul**: Fixed code examples in the README to match actual API signatures and documented 5 previously unlisted features (ProjectionAdapter, MigrationTrainer, TaskArithmetic, VsaAdapter, Slerp/Reject).
- **API Reference**: Rewrote structural API documentation covering all 12 classes and types.
- **New Guides**: Added `10-projection-migration.md`, `11-task-arithmetic.md`, and `12-vsa.md`.
- **Trainer Guides**: Corrected TripletTrainer / InfoNCETrainer examples to match current API signatures.

### 🔧 Build & Tooling
- **Centralized Version Control**: Added `scripts/sync-versions.ts` to automatically sync versions from the root `package.json` to all sub-packages.
- **DTS Build Fixes**: Added `tsconfig.build.json` to sub-packages to resolve TS compilation `rootDir` errors.
- **CI Enhancements**: Ensured AssemblyScript compilation occurs before running tests.
- **Publish Workflow**: Enabled npm Trusted Publishing (OIDC) and upgraded `setup-bun` to v2.

---

## 🇯🇵 日本語リリースノート

### ✨ ハイライト

ドキュメントと実装の整合性を全面的に修正し、ビルドパイプラインを改善しました。

### 📝 ドキュメント

- **README**: 全コード例を実際の API シグネチャに修正、未記載の5機能（`ProjectionAdapter`, `MigrationTrainer`, `TaskArithmetic`, `VsaAdapter`, `Slerp`/`Reject`）を追加。
- **API リファレンス**: 全12クラス・全型定義を網羅する形で全面書き換え。
- **新規ガイド**: `10-projection-migration.md`, `11-task-arithmetic.md`, `12-vsa.md` を追加。
- **Trainer ガイド**: TripletTrainer / InfoNCETrainer のコード例を現行 API に修正。

### 🔧 ビルド & ツーリング

- **バージョン一元管理**: `scripts/sync-versions.ts` でルートの `package.json` から全サブパッケージへ自動同期。
- **DTS ビルド修正**: 各パッケージに `tsconfig.build.json` を追加し、`paths` 起因の `rootDir` エラーを解消。
- **CI 改善**: WASM ビルドが確実にテスト前に実行されるように修正。
- **Publish ワークフロー**: npm Trusted Publishing (OIDC) 対応、`setup-bun` v2 に更新。
