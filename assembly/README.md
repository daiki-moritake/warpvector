# assembly/ — AssemblyScript WASM Source

このディレクトリには、WarpVector の WASM アクセラレーション用 [AssemblyScript](https://www.assemblyscript.org/) ソースコードが含まれています。

## ビルドフロー

```
assembly/index.ts
    ↓  (AssemblyScript compiler)
build/optimized.wasm
    ↓  (Base64 エンコード)
packages/core/src/wasm/wasm-binary.ts
```

### ビルドコマンド

```bash
bun run build:wasm
```

内部的には `scripts/build-wasm.ts` が以下を実行します：

1. `bunx asc assembly/index.ts -o build/optimized.wasm -O3 --noAssert --enable simd`
2. コンパイル済み WASM バイナリを Base64 文字列に変換
3. `packages/core/src/wasm/wasm-binary.ts` にインライン展開

## 設計意図

- **ゼロ依存**: WASM バイナリを TypeScript ファイル内にインライン化することで、外部ファイル読み込みなしにランタイムで WASM を初期化
- **エッジ対応**: Cloudflare Workers 等の制約環境でもファイルシステムアクセス不要
- **SIMD 最適化**: ベクトル演算を SIMD 命令で高速化
