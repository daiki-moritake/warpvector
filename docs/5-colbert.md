# Late Interaction / ColBERT

In general vector search, the meaning of an entire document or an entire query is compressed into a "single vector" (e.g., 1536 dimensions) for comparison (Dense Retrieval). However, this poses a challenge because specific "subtle nuances" and "word combinations" contained in long documents get squashed during the compression process.

WarpVector's `ColbertAdapter` is an adapter that enables **Late Interaction (ColBERT Architecture)** in edge environments.
Instead of treating queries and documents as a single vector, they are maintained as a "set of vectors (matrix) for each token (word)", and a precise brute-force matching (MaxSim) is performed at the very end of the search.

## 1. How Late Interaction (MaxSim) Works

1. Represent the query "A B C" as multiple vectors like `[Vec_A, Vec_B, Vec_C]`.
2. Similarly, represent the document "X Y Z ..." as `[Vec_X, Vec_Y, Vec_Z, ...]`.
3. For each token in the query, calculate the similarity with all tokens in the document, and find the **"highest similarity (MaxSim)"**.
4. The sum of the MaxSim for all query tokens becomes the final document score.

With this, if the "concept of A" exists somewhere in the document, and the "concept of B" exists somewhere else, they are accurately matched. It provides a highly accurate search experience that is absolutely impossible with a single vector.

## 2. Ultra-Acceleration with WASM

This "brute-force matching" has a fatal flaw: executing it in TypeScript/JavaScript is hopelessly slow (because it requires multiple nested loops and tens of thousands of dot products).
WarpVector dramatically accelerates this MaxSim calculation by running it on the flat memory of **WebAssembly (WASM)**, utilizing loop unrolling and SIMD-like processing.

## 3. Basic Usage

```typescript
import { ColbertAdapter } from 'warpvector';

const adapter = new ColbertAdapter();

// Queries and documents should be prepared as "flattened arrays of token vectors"
// Example: If you have 5 tokens of 32 dimensions, it's a 160-element Float32Array (32 * 5 = 160)
const queryTokens = getQueryTokenMatrix(); 

const doc1Tokens = getDocTokenMatrix(doc1);
const doc2Tokens = getDocTokenMatrix(doc2);
const doc3Tokens = getDocTokenMatrix(doc3);

const documents = [doc1Tokens, doc2Tokens, doc3Tokens];
const dimension = 32; // Dimensionality per token

// Calculate MaxSim against all documents ultra-fast on WASM, and rank them by score
const rankedResults = adapter.rank(queryTokens, documents, dimension);

console.log(rankedResults);
// Example Output:
// [
//   { index: 1, score: 12.45 }, // doc2 is 1st
//   { index: 0, score:  9.12 }, // doc1 is 2nd
//   { index: 2, score:  4.33 }  // doc3 is 3rd
// ]
```

## 4. Use Cases

Late Interaction via the ColBERT architecture shows overwhelming power, especially in **RAG (Retrieval-Augmented Generation)**.

For long, complex questions from users, or queries involving multiple conditions like "Tell me about [Topic A] from the perspective of [Topic B]", single-vector searches tend to pull up "other documents that are vaguely similar overall". However, by using ColBERT, you can precisely pinpoint and retrieve documents where both the [Topic A] and [Topic B] tokens appear in the text.
