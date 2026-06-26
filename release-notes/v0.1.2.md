# 🌌 v0.1.2 — Security & Memory Safety Bug Fixes / セキュリティとメモリ安全性の修正

In this release, we solved critical bugs related to WASM memory management, enhanced SQL injection prevention in Prisma integration, and optimized dynamic quantization and pgvector binary output formats.

今回のリリースでは、WASMメモリ管理のバグ修正、SQLインジェクション対策、動的量子化およびpgvectorバイナリ出力の改善など、安全性とパフォーマンス、精度の向上に関する重大な修正を行いました。

---

## 🇬🇧 English Release Notes

### 🛠 Key Changes & Fixes

- **WASM Memory Safety & Dynamic Allocation**:
  - Introduced a stack-like allocator to prevent memory collisions when multiple adapters are chained.
  - Calculated intermediate buffers dynamically in `MlpAdapter` to prevent buffer overflows when working with high-dimensional vectors (e.g., 1536d or 3072d).
  - Swapped pointer handling in `ColbertAdapter` to stack-based allocation, avoiding memory corruption of system regions (around address `0`).
- **Performance Optimizations**:
  - Enabled SIMD compiler optimizations during AssemblyScript compilation.
  - Ported the Adam weight updates loop in `BaseTrainer` to WASM, preventing main thread freezing (UI blocks) and improving training speed.
- **Enhanced Security (SQL Injection Prevention)**:
  - Implemented validation and sanitization for table names, columns, where clauses, and limit values inside the Prisma integration, eliminating SQL injection vulnerabilities.
- **Quantization & Database Integration**:
  - Added "Dynamic Calibration Quantization" in `QuantizationAdapter` to automatically calculate scaling factors from vector dynamic range, improving inner-product reconstruction accuracy.
  - Implemented binary formatting to return a binary bit-string (e.g. `'1101...'`) suitable for pgvector `bit(N)` fields when binary-quantized vectors are processed.

---

## 🇯🇵 日本語リリースノート

### 🛠 主な修正・変更内容

* **WASMメモリ安全性の向上と動的アロケーション**:
  * 簡易メモリアロケータを導入し、複数アダプター連結時におけるWASMメモリの衝突を防止。
  * `MlpAdapter` の中間バッファを動的に算出し、1536次元や3072次元などの高次元ベクトルでのバッファオーバーフローを解消。
  * `ColbertAdapter` の一時ポインタ処理をスタック型アロケーションに変更し、WASMのシステム領域（`0`番地付近）の破壊を防止。
* **パフォーマンス最適化**:
  * WASMコンパイル時の SIMD 最適化を有効化。
  * 学習エンジン（`BaseTrainer`）の Adam パラメータ更新ループを WASM 側に移植し、学習の高速化とメインスレッドのフリーズ（画面ブロック）を防止。
* **セキュリティの強化 (SQLインジェクション対策)**:
  * Prisma 統合におけるテーブル名・カラム名・where句・limit値のバリデーションとサニタイズを徹底し、SQL インジェクション脆弱性を完全に排除。
* **量子化とDB連携の改善**:
  * `QuantizationAdapter` にて、ベクトルのダイナミックレンジからスケールを自動算出する「動的キャリブレーション量子化」を追加。内積復元精度が劇的に向上。
  * 二値化された `Uint8Array` ベクトルが渡された際に、pgvector の `bit(N)` 型に適した2進数ビット文字列（例：`'1101...'`）を返すフォーマット処理を実装。
