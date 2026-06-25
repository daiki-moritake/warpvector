# 🌌 WarpVector v0.3.1 — Performance Optimizations & Allocation Reduction / パフォーマンス最適化 & アロケーション極限削減

WarpVector `v0.3.1` has been released. This update focuses on optimizing WASM/SIMD core computational loops and eliminating redundant object allocations during JS / WASM boundary data transfers, substantially increasing batch processing throughput.

---

## 🇬🇧 English Release Notes

### ✨ Highlights

#### 🧠 Loop Cache & Instruction Parallelization in WASM Core
We inverted the dimension and batch loops inside `tuneBatchWasm`.
- **Contiguous Writes**: Output writes are now completely sequential, minimizing CPU cache misses due to memory strides.
- **Instruction-Level Parallelism (ILP)**: We split the dot product (`innerProductSimd`) accumulator into 4 independent SIMD registers. This mitigates data-dependency stalls in the CPU pipeline.

#### 📊 Allocation Reduction in JS / WASM Data Copying
- Eliminated redundant `new Float32Array` (WASM shared memory views) heap allocations inside hot batch loops (`IntentAdapter`, `ColbertAdapter`, and `MlpAdapter`) by caching and reusing a single view.
- Replaced manual JS copying loops with optimized `Float32Array.prototype.slice()` and `set()`, which map directly to high-speed native `memcpy` operations.
- This dramatically reduces Garbage Collection (GC) pressure under high-concurrency production workloads.

---

### 📈 Performance Improvements

Measured using 1536-dimensional vectors and a batch size of 10,000.

| Metric | v0.3.0 (Before) | v0.3.1 (After) | Improvement |
|------|------|------|------|
| **WASM batch tune (10000 vecs, 1536d)** | 2355.87 ms (4,245 vecs/s) | **2022.53 ms (4,944 vecs/s)** | **+16.4% Throughput** |
| **Int8 Scalar Quantization** | 314.90 ms (317,564 vecs/s) | **268.15 ms (372,924 vecs/s)** | **+17.3% Performance** |
| **Binary (1-bit) Quantization** | 90.17 ms | **82.39 ms** | **+8.6% Performance** |

---

### 🧪 Tests
- **290 tests** / 43 files / **1,562 expect()** — All Passed ✅
- Verified mathematical equivalence between batch and single operations, and ensured zero memory leaks (no WASM memory offset growth over consecutive executions).

---

### ⬆️ How to Upgrade

```bash
npm install warpvector@0.3.1
```
**No breaking changes.** Fully compatible with v0.3.0 / v0.2.0.

---

## 🇯🇵 日本語リリースノート

### ✨ ハイライト

#### 🧠 WASM コア演算のキャッシュ & 命令並列化の最適化
バッチアフィン変換を行う `tuneBatchWasm` の次元ループとバッチループの順序を反転させました。
- **連続書き込み**: メモリへの書き込みをシーケンシャルに行うことで、書き込みストライドによる CPU キャッシュミスを最小化。
- **レジスタ並列性の向上 (ILP)**: 内積計算（`innerProductSimd`）のアキュムレータを4つの独立した SIMD レジスタに分割し、CPU命令パイプラインのデータ依存ストールを排除しました。

#### 📊 JS / WASM データコピーの極小化 (アロケーション削減)
- 大規模バッチ処理ループ（`IntentAdapter`、`ColbertAdapter`、`MlpAdapter`）で毎回発生していた `new Float32Array`（WASM共有メモリビュー）のヒープアロケーションをループ外に追い出し、単一ビューを使い回す設計に最適化しました。
- 結果の抽出および書き込みにおいて、JSループによるコピー処理を廃止し、最適化された `Float32Array.prototype.slice()` / `set()` による高速な `memcpy` 処理へと刷新。
- これにより、本番環境での高コンカレンシー下におけるガベージコレクション（GC）負荷を極限まで低減しています。

---

### 📈 パフォーマンス改善データ

1536次元ベクトル、10,000件のバッチを用いて最適化前後のパフォーマンスを計測しました。

| 項目 | v0.3.0 (最適化前) | v0.3.1 (最適化後) | 改善率 |
|------|------|------|------|
| **WASM batch tune (10000 vecs, 1536d)** | 2355.87 ms (4,245 vecs/s) | **2022.53 ms (4,944 vecs/s)** | **+16.4% スループット向上** |
| **Int8 Scalar Quantization** | 314.90 ms (317,564 vecs/s) | **268.15 ms (372,924 vecs/s)** | **+17.3% 処理速度向上** |
| **Binary (1-bit) Quantization** | 90.17 ms | **82.39 ms** | **+8.6% 処理速度向上** |

---

### 🧪 テスト検証
- **290 テスト** / 43 ファイル / **1,562 expect()** — 全パス ✅
- アフィン変換の数学的一致、バッチ処理と単一処理の等価性、メモリリークがないこと（consecutive execution without WASM memory offset growth）を徹底して検証済みです。

---

### ⬆️ アップグレード方法

```bash
npm install warpvector@0.3.1
```
**破壊的変更はありません。** v0.3.0 / v0.2.0 からの完全なドロップインアップグレードが可能です。
