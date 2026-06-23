# Ecosystem Integrations

While WarpVector is extremely powerful as a standalone library, it is designed to integrate seamlessly with the existing ecosystem for developing LLM applications (LangChain, LlamaIndex, Prisma).
This allows you to reap the benefits of WarpVector's dynamic affine transformations and quantization with almost no modifications to your existing codebase.

## 1. LangChain Integration (`WarpEmbeddings`)

We provide a class that wraps LangChain's `Embeddings` interface.
It uses the original embeddings as-is when saving to a VectorStore (during Document creation), but **dynamically warps the space through WarpVector's adapter only during searches (Querying)**.

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { IntentAdapter } from "warpvector";
import { WarpEmbeddings } from "warpvector/integrations/langchain";

// 1. Prepare standard Embeddings and a WarpVector adapter
const baseEmbeddings = new OpenAIEmbeddings();
const adapter = new IntentAdapter(myIntents); // Pre-defined intents

// 2. Wrap them to create WarpEmbeddings
const warpEmbeddings = new WarpEmbeddings({
  baseEmbeddings: baseEmbeddings,
  adapter: adapter,
  intentName: "riskAnalysis" // The name of the intent you want to apply
});

// 3. Just pass it directly to an existing VectorStore!
const vectorStore = new MemoryVectorStore(warpEmbeddings);

// (When saving documents, they are saved using baseEmbeddings)
await vectorStore.addDocuments([...]);

// (When searching, it automatically searches using the query vector warped by the "riskAnalysis" intent)
const results = await vectorStore.similaritySearch("Market crash", 5);

// If you want to switch the intent dynamically
warpEmbeddings.setIntent("economicImpact");
const results2 = await vectorStore.similaritySearch("Market crash", 5);
```

## 2. LlamaIndex Integration (`WarpLlamaIndexEmbeddings`)

Similar to LangChain, it fully supports the `BaseEmbedding` interface of LlamaIndex (TS Version).
You can pass it directly to `VectorStoreIndex` or `Retriever`.

```typescript
import { OpenAIEmbedding, VectorStoreIndex } from "llamaindex";
import { IntentAdapter } from "warpvector";
import { WarpLlamaIndexEmbeddings } from "warpvector/integrations/llama-index";

const baseEmbeddings = new OpenAIEmbedding();
const adapter = new IntentAdapter(myIntents);

// Create a wrapper for LlamaIndex
const warpLlamaIndexEmbeddings = new WarpLlamaIndexEmbeddings({
  baseEmbeddings: baseEmbeddings,
  adapter: adapter,
  intentName: "legalAnalysis"
});

// Directly integrate it into LlamaIndex's index generation and query engine
const index = await VectorStoreIndex.fromDocuments(documents, {
  serviceContext: { embedModel: warpLlamaIndexEmbeddings }
});

const queryEngine = index.asQueryEngine();
const response = await queryEngine.query("What are the conditions for contract cancellation?");
```

## 3. Prisma + pgvector Integration (`withWarpVector`)

If you are using PostgreSQL's `pgvector` extension alongside Prisma, you can **complete WarpVector vector inference and database search entirely within Prisma Client methods** without writing any raw SQL. This utilizes the Prisma Client Extension mechanism.

```typescript
import { PrismaClient } from '@prisma/client';
import sql from 'sql-template-tag';
import { WhiteningAdapter } from 'warpvector';
import { withWarpVector } from 'warpvector/integrations/prisma';

// Example: Online PCA Adapter
const adapter = new WhiteningAdapter(1536, { numComponents: 1 });

// Attach the WarpVector extension to Prisma Client
const prisma = new PrismaClient().$extends(
  withWarpVector({
    adapter: adapter,
    vectorField: "embedding", // Column name where vectors are stored in the Prisma schema
    distanceOperator: "<=>",  // Cosine Distance
    // intentName: "myIntent" // You can specify the intent here when using IntentAdapter, etc.
  })
);

// The extension method searchByVector is now available!
// Just pass a raw vector (exactly as obtained from the API),
// and internally adapter.tune() is executed automatically, which is then expanded into pgvector's SQL.
const results = await prisma.document.searchByVector({
  vector: rawSearchVector,
  topK: 10,
  where: sql`category = 'science'` // Optional: Use sql-template-tag for safe WHERE clauses
});

console.log(results); // Returns IDs, scores, and record info
```
