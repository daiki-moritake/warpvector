# 15. Time-Reversal Wave Reranker

`TimeReversalReranker` is a re-ranking module introducing a completely new paradigm, applying the principles of a "Time-Reversal Mirror" from physics to vector search.

It solves the problem in RAG (Retrieval-Augmented Generation) and elsewhere where "too many similar documents are hit, making it impossible to identify the single document (the true source) that was actually needed."

## Concept: Why Time Reversal?

In typical vector search, the "intent" of the query **diffuses** across multiple documents in the embedding space. As a result, not only the "source document" you truly wanted to find, but also surrounding "documents with coincidentally similar contexts" end up with high scores.

The Time-Reversal Reranker solves this through the following process:

1. **Medium Observation**: Calculates the mutual similarity among the top documents hit by the search (e.g., Top-50) and builds a network (manifold) of "how the meaning diffuses."
2. **Initial Wavefront Observation**: Treats the similarity between the query and each document as the "amplitude of the observed wave."
3. **Time-Reversal / Rewind**: "Rewinds" the wave on the constructed network using a Graph Laplacian (Inverse Graph Diffusion).
4. **Focusing**: The scores of surrounding documents that gained points through diffusion sharply decay, while the score strongly concentrates (sharpens) on the **"true latent document"** that was the source of the wave.

---

## Usage

Unlike normal `WarpAdapter`s, the reranker executes on a pair of "Query" and "Candidate Group".
It is used as a post-processing step after performing a Top-K search in a Vector DB. Particularly, if the Vector DB has already returned scores (cosine similarity), you can omit unnecessary recalculations by passing them via `initialScores`.

```typescript
import { TimeReversalReranker } from 'warpvector';

// Initialize the reranker
const reranker = new TimeReversalReranker({
  tau: 1.5,             // Strength of time reversal (sharpness)
  threshold: 0.1,       // Similarity threshold to make the graph between candidates sparser
  normalizeGraph: true, // Graph normalization to prevent the hubness problem (Default: true)
  iterations: 3         // Number of iterations (Default: 1)
});

// A group of top candidate vectors retrieved from a Vector DB, etc.
const candidates = [
  new Float32Array([...]), // doc 1
  new Float32Array([...]), // doc 2
  // ...
];

// Pattern A: When passing a query vector and having it calculate the scores
const queryVector = new Float32Array([...]);
const resultsA = reranker.rerank(queryVector, candidates);

// Pattern B: When reusing the search scores from the DB side (Recommended/Fastest)
const initialScores = [0.95, 0.82, ...]; 
const resultsB = reranker.rerank(null, candidates, initialScores);

/*
results is sorted in descending order of score in the following format:
[
  { 
    originalIndex: 1, 
    score: 0.99,         // Sharp score after time reversal
    initialScore: 0.82,  // Initial cosine similarity
    vector: Float32Array 
  },
  ...
]
*/
```

---

## Parameter Tuning

### `tau` (Strength of Time Reversal)
- Default is `1.0`.
- **Increase (e.g., 2.0 ~ 5.0)**: Sharpness becomes stronger. The score gap between the source and surrounding documents widens dramatically. However, if set too large, numbers may diverge, risking the amplification of noise (incorrect sources).
- **Decrease (e.g., 0.1 ~ 0.5)**: Results in a milder correction.

### `iterations` (Number of Iterations)
- Default is `1`.
- If `tau` is too large, overshoot (score inversion) may occur. To prevent this, setting a smaller `tau` (e.g., `0.5`) and increasing `iterations` to `3` allows the focus to converge more smoothly and safely.

### `normalizeGraph` (Graph Normalization)
- Default is `true`.
- Performs normalization using the Random Walk Laplacian. This suppresses the phenomenon where documents that are "similar to many other documents (hubs)" unfairly siphon up high scores (the hubness problem), stabilizing the calculation. Leave it as `true` unless you have a specific reason not to.

### `threshold` (Edge Cutoff Threshold)
- Default is `0.0`.
- If the cosine similarity between candidates is less than this value, they are considered "unconnected (waves do not diffuse)" on the graph.
- Setting it to `0.1` or `0.2` is often effective to cut out minute similarities that act as noise, thereby stabilizing the calculation.

---

## Differences from `SoftWhiteningAdapter`

WarpVector has a similarly named `SoftWhiteningAdapter`.

- **`SoftWhiteningAdapter`**: 
  - Resolves the "spatial bias" of the entire model via **streaming learning** using time reversal (Inverse Heat Equation).
  - It is a "Pre-processing" step that sharpens a single query vector beforehand.
- **`TimeReversalReranker`**:
  - Rewinds the "semantic diffusion" among the hit documents using a **local document graph**.
  - It is a "Post-processing / Reranking" step that sharpens the search results.

In a production RAG system, **combining these** (sharpening the query with the adapter, then identifying the source with the reranker) enables extremely precise document identification.
