# 🌌 v0.1.1 — WarpVector is Live! / WarpVector リリース開始

The first official release of **WarpVector** is finally here! WarpVector provides a blazing fast, zero-dependency solution to warp and adapt embeddings right at the edge.

WarpVectorの最初の公式リリースがついに登場しました！WarpVectorは、エッジ環境で直接埋め込みベクトルを動的に歪め、適応させるための極めて高速でゼロ依存のソリューションを提供します。

---

## 🇬🇧 English Release Notes

WarpVector provides a blazing fast, zero-dependency solution to warp and adapt embeddings right at the edge. Heavy math operations are offloaded to embedded WebAssembly.

### ✨ Highlights
- **Performance First:** Heavy math operations (matrix multiplication, SGD+Momentum training) are offloaded to embedded WebAssembly (SIMD).
- **LoRA Support:** Efficiently perform affine transformations on high-dimensional embeddings using Low-Rank Adaptation.
- **Universal Compatibility:** Works everywhere. No extra `.wasm` files to serve — it's completely inlined and bundled.

### 🚀 Get Started
```bash
npm install warpvector
# or
bun add warpvector
```

---

## 🇯🇵 日本語リリースノート

WarpVector は、エッジ環境で直接埋め込み（Embedding）を動的に変形・適応させるための超高速かつゼロ依存関係のソリューションを提供します。重い数学演算は組み込みの WebAssembly にオフロードされます。

### ✨ ハイライト
- **パフォーマンス第一主義**: 行列乗算や SGD+Momentum 学習などの重い演算処理を、組み込みの WebAssembly (SIMD) にオフロードします。
- **LoRA (低ランク適応) サポート**: 低ランク適応 (Low-Rank Adaptation) を用いて、高次元の埋め込みベクトルに対して効率的なアフィン変換を実行します。
- **ユニバーサルな互換性**: あらゆる環境で動作します。外部 of `.wasm` ファイルを配信する必要はなく、完全にインライン化され、単一のパッケージとしてバンドルされています。

### 🚀 はじめに
```bash
npm install warpvector
```
