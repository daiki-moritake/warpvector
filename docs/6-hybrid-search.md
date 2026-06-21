# Hybrid Search Fusion

The best practice for modern search systems is to combine "Vector Search (Dense Search: Semantic Similarity)" like WarpVector with "Keyword Search (Sparse Search/BM25: Exact Word Matching)" like Elasticsearch.
This is known as **Hybrid Search**.

WarpVector provides independent fusion utilities to "fairly combine" the scores from two completely different search engines.

## 1. RRF (Reciprocal Rank Fusion)

RRF is an algorithm that integrates results by **completely ignoring the absolute values (raw numbers) of the scores** obtained from different search systems and **using only their "Rank" (ordinal position)**.
It is most effective when combining systems with completely different scales, such as vector search scores (0.0 to 1.0) and keyword search scores (10 to 150, etc.).

$$ RRF\_Score = \sum \frac{1}{k + Rank} $$
* Generally, the constant $k = 60$ is widely used.

### Usage

```typescript
import { rrf } from 'warpvector';

// Results from a Vector DB (e.g., Pinecone)
const denseResults = [
  { id: "docA", score: 0.95 }, // 1st
  { id: "docB", score: 0.88 }, // 2nd
  { id: "docC", score: 0.72 }  // 3rd
];

// Results from a Keyword DB (e.g., Elasticsearch)
const sparseResults = [
  { id: "docB", score: 45.2 }, // 1st
  { id: "docD", score: 32.1 }, // 2nd
  { id: "docA", score: 15.0 }  // 3rd
];

// Just pass an array of lists, and it returns a new rank-based fused list
const fusedResults = rrf([denseResults, sparseResults]);

console.log(fusedResults);
// docB and docA will rank at the top (since they are highly rated by both systems)
```

## 2. RSF (Relative Score Fusion)

RSF is used when you want to emphasize the **"relative magnitude of the scores"** rather than the rank.
It performs Min-Max normalization (converting scores to 0.0 ~ 1.0) on the scores within the result list of each system, multiplies them by a specified weight, and then adds them together.

This allows for fine-tuning such as, "I want to heavily emphasize the semantic match of the vector search (70%), but also factor in exact keyword matches slightly (30%)."

### Usage

```typescript
import { rsf } from 'warpvector';

// 1st Argument: An array of result lists
// 2nd Argument: An array of weights for each list
const fusedResults = rsf(
  [denseResults, sparseResults], 
  [0.7, 0.3] // Fuse with a weight of 70% for Dense and 30% for Sparse
);
```

## Metadata Integration

Both `rrf` and `rsf` automatically carry over any `metadata` properties possessed by each item. If an item with the same ID exists in multiple lists and holds different metadata in each, they are merged via a Shallow Merge and included in the final result.
