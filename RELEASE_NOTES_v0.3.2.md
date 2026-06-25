# 🌌 WarpVector v0.3.2 — Pipeline Deserialization Support & CLI Version Sync / パイプライン復元サポート & CLIバージョン同期

WarpVector `v0.3.2` has been released. This update addresses critical deserialization errors in pipelines containing security/safety steps, resolves a mismatch in CLI version reporting, and corrects invalid examples in the documentation.

---

## 🇬🇧 English Release Notes

### ✨ Highlights

#### 🧠 Pipeline Deserialization Support for Safety & Quantization
We resolved an issue where `WarpPipeline.importState` would fail with an `Unknown adapter type` error if the pipeline state contained security/safety adapters.
- **Auto-registration**: `SafeQuantizationAdapter` and `AnomalyDetectionAdapter` are now automatically registered to the `WarpPipeline` adapter registry upon package load.
- **Robustness**: Ensured that security-wrapped pipelines can be fully serialized and restored across server restarts or edge worker instances.

#### 📊 CLI Version Auto-sync in `create-warpvector-app`
- Replaced the hardcoded `'0.2.0'` CLI tool version string inside `create-warpvector-app` with a dynamic version loader reading from its `package.json`.
- This ensures CLI tools always display correct, up-to-date versions corresponding to the latest release package (`v0.3.2`).

#### 📖 Documentation Correction
- Fixed an invalid example in `getting-started.md` that incorrectly referenced the non-existent `.quantize("int8")` method.
- Updated the example to use the standard, correct `.setFinalStage()` pattern utilizing the `QuantizationAdapter`.

---

## 🇯🇵 日本語リリースノート

### ✨ ハイライト

#### 🧠 安全性・量子化アダプターのパイプライン復元（デシリアライズ）サポート
セキュリティおよび入力検証用のアダプターがパイプライン状態に含まれている場合、`WarpPipeline.importState` によるデシリアライズ処理が `Unknown adapter type` エラーで失敗する問題を解消しました。
- **自動登録の追加**: パッケージのロード時に `SafeQuantizationAdapter` および `AnomalyDetectionAdapter` が `WarpPipeline` のアダプターレジストリへ自動登録されるようになりました。
- **堅牢性の向上**: セキュリティフィルターや異常値検知を挟んだ高度な変換パイプライン全体を、サーバー再起動やエッジワーカーの別インスタンスへ安全にシリアライズ・復元できます。

#### 📊 `create-warpvector-app` CLI のバージョン自動同期
- `create-warpvector-app` 内部にハードコードされていたバージョン文字列 `'0.2.0'` を廃止し、自身の `package.json` から動的にバージョンを読み込むように改善しました。
- これにより、CLI ツールの出力するバージョン表記が、リリースされる最新のパッケージバージョン（`v0.3.2`）と常に自動的に同期され、利用者の混乱を防止します。

#### 📖 ドキュメントの誤記修正
- `getting-started.md` のサンプルコード内に存在した、未実装の `.quantize("int8")` を呼び出している箇所を修正しました。
- 代わりに、正しい標準 API である `.setFinalStage("QuantizationAdapter", new QuantizationAdapter({ type: "int8", dim: 3 }))` を用いる記述に修正しました。

---

### 🧪 テスト検証 / Tests
- **291 tests** / 43 files / **1,567 expect()** — All Passed ✅
- Added end-to-end integration tests to verify the serialization and deserialization flow for safety/quantization-wrapped pipelines.

---

### ⬆️ アップグレード方法 / How to Upgrade

```bash
npm install warpvector@0.3.2
# または / or
bun add warpvector@0.3.2
```
**No breaking changes.** Fully compatible with v0.3.x.
**破壊的変更はありません。** v0.3.x 系統からそのまま安全にアップグレード可能です。
