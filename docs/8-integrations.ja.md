# エコシステム統合 (Integrations)

WarpVector は単独のライブラリとしても非常に強力ですが、既存の LLM アプリケーション開発エコシステム（LangChain, LlamaIndex, Prisma）とシームレスに連携できるように設計されています。
これにより、既存のコードベースをほとんど変更することなく、WarpVector の動的アフィン変換や量子化の恩恵を受けることができます。

## 1. LangChain 統合 (`WarpEmbeddings`)

LangChain の `Embeddings` インターフェースをラップ（包み込む）するクラスを提供しています。
VectorStore への保存時（Document生成時）は元の埋め込みをそのまま使い、**検索時（Query時）にのみ WarpVector のアダプタを通して空間を動的にワープ**させます。

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { IntentAdapter } from "warpvector";
import { WarpEmbeddings } from "warpvector/integrations/langchain";

// 1. 通常の Embeddings と WarpVector のアダプタを用意
const baseEmbeddings = new OpenAIEmbeddings();
const adapter = new IntentAdapter(myIntents); // 事前に定義した意図

// 2. ラップして WarpEmbeddings を作成
const warpEmbeddings = new WarpEmbeddings({
  baseEmbeddings: baseEmbeddings,
  adapter: adapter,
  intentName: "riskAnalysis" // 適用したい意図の名前
});

// 3. 既存の VectorStore にそのまま渡すだけ！
const vectorStore = new MemoryVectorStore(warpEmbeddings);

// (ドキュメント保存時は baseEmbeddings のまま保存される)
await vectorStore.addDocuments([...]);

// (検索時は自動的に "riskAnalysis" の意図でワープされたクエリベクトルで検索される)
const results = await vectorStore.similaritySearch("Market crash", 5);

// 意図を動的に切り替えたい場合
warpEmbeddings.setIntent("economicImpact");
const results2 = await vectorStore.similaritySearch("Market crash", 5);
```

## 2. LlamaIndex 統合 (`WarpLlamaIndexEmbeddings`)

LangChain と同様に、LlamaIndex (TS版) の `BaseEmbedding` インターフェースも完全にサポートしています。
`VectorStoreIndex` や `Retriever` にそのまま渡すことができます。

```typescript
import { OpenAIEmbedding, VectorStoreIndex } from "llamaindex";
import { IntentAdapter } from "warpvector";
import { WarpLlamaIndexEmbeddings } from "warpvector/integrations/llama-index";

const baseEmbeddings = new OpenAIEmbedding();
const adapter = new IntentAdapter(myIntents);

// LlamaIndex 用のラッパーを作成
const warpLlamaIndexEmbeddings = new WarpLlamaIndexEmbeddings({
  baseEmbeddings: baseEmbeddings,
  adapter: adapter,
  intentName: "legalAnalysis"
});

// LlamaIndex のインデックス生成・クエリエンジンにそのまま組み込む
const index = await VectorStoreIndex.fromDocuments(documents, {
  serviceContext: { embedModel: warpLlamaIndexEmbeddings }
});

const queryEngine = index.asQueryEngine();
const response = await queryEngine.query("契約解除の条件は？");
```

## 3. Prisma + pgvector 統合 (`withWarpVector`)

PostgreSQL の `pgvector` 拡張と Prisma を使用している場合、SQLを直接書くことなく、**WarpVector によるベクトル推論とデータベース検索を Prisma Client メソッドの中で完結** させることができます。Prisma Client Extension の仕組みを利用しています。

```typescript
import { PrismaClient } from '@prisma/client';
import { WhiteningAdapter } from 'warpvector';
import { withWarpVector } from 'warpvector/integrations/prisma';

// 例: オンラインPCAアダプタ
const adapter = new WhiteningAdapter(1536, { numComponents: 1 });

// Prisma Client に WarpVector 拡張をアタッチ
const prisma = new PrismaClient().$extends(
  withWarpVector({
    adapter: adapter,
    vectorField: "embedding", // Prisma schema 上のベクトル保存先カラム名
    distanceOperator: "<=>",  // コサイン距離
    // intentName: "myIntent" // IntentAdapter等の場合はここで意図を指定可能
  })
);

// 拡張メソッド searchByVector が生える！
// 生のベクトル(APIから取得したままのベクトル)を渡すだけで、
// 内部で adapter.tune() が自動実行され、その後 pgvector の SQL に展開されます。
const results = await prisma.document.searchByVector({
  vector: rawSearchVector,
  topK: 10,
  where: "category = 'science'" // オプション: 通常の Prisma WHERE 句も記述可能
});

console.log(results); // IDやスコア、レコード情報が返ります
```
