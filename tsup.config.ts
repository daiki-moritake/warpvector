import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm"],
  dts: { resolve: [/^@warpvector\//] },
  clean: true,
  // @warpvector/* サブパッケージを dist にバンドル（新規追加時も自動対応）
  noExternal: [/^@warpvector\//],
  // peerDependencies は外部モジュールのままにする
  external: ["@prisma/client", "@langchain/core", "sql-template-tag"],
});

