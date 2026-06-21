import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    prisma: "src/prisma.ts",
    langchain: "src/langchain.ts",
  },
  format: ["cjs", "esm"],
  dts: { resolve: [/^@warpvector\//] },
  clean: true,
  // @warpvector/* サブパッケージを dist にバンドル（新規追加時も自動対応）
  noExternal: [/^@warpvector\//],
  // peerDependencies は外部モジュールのままにする
  external: ["@prisma/client", "@langchain/core", "sql-template-tag"],
});
