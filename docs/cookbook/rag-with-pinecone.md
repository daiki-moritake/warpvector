# Cookbook: Cost-Efficient RAG with Pinecone

Storing millions of vectors in a managed vector database like Pinecone or Qdrant can quickly become expensive, as these databases keep vectors in memory. 

WarpVector allows you to compress `Float32` vectors into `Int8` or `Binary` formats *before* sending them to your database, reducing storage and memory costs by up to 96%.

## Implementation

```typescript
import { Pinecone } from '@pinecone-database/pinecone';
import { QuantizationAdapter } from 'warpvector/extras';

// 1. Initialize the Binary Quantization Adapter
const quantizer = new QuantizationAdapter({ type: 'binary', dim: 1536 });
const pc = new Pinecone();
const index = pc.index('my-rag-index');

/**
 * INDEXING DOCUMENTS
 */
async function indexDocument(id: string, text: string) {
  // Get standard Float32 embedding from OpenAI
  const rawVector = await getOpenAIEmbedding(text);
  
  // Compress to Binary (1536 float32 -> 1536 bits = 192 bytes)
  const compressedVector = quantizer.tune(rawVector);
  
  // Store in Pinecone
  await index.upsert([{
    id,
    values: Array.from(compressedVector), // Convert Uint8Array to regular array
    metadata: { text }
  }]);
}

/**
 * SEARCHING (RAG)
 */
async function search(query: string) {
  const rawQueryVector = await getOpenAIEmbedding(query);
  
  // Compress the query vector using the same quantizer
  const compressedQueryVector = quantizer.tune(rawQueryVector);
  
  // Search Pinecone using the compressed vector
  // Make sure your Pinecone index is configured for hamming distance or dot product
  const results = await index.query({
    vector: Array.from(compressedQueryVector),
    topK: 5,
    includeMetadata: true
  });
  
  return results;
}
```

## ROI (Return on Investment)
- **Float32**: 1,000,000 vectors × 1536 dims × 4 bytes = ~6.14 GB of RAM.
- **Binary**: 1,000,000 vectors × 1536 dims × 0.125 bytes = ~192 MB of RAM.

You save ~96.8% in database storage, which directly translates to significantly lower monthly infrastructure bills.
