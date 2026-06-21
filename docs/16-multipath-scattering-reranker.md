# 16. Multipath Scattering Reranker

`MultipathScatteringReranker` is a re-ranking module introducing a completely new paradigm inspired by the "Multipath Scattering Theory of Waves" from physics.

In vector search, it eliminates "isolated accidental noise" and identifies "true source hubs connected through (and supported by multiple reflections from) many relevant documents."

## Concept: Why Multipath Scattering?

The top group of searched documents forms a semantic "network (scattering medium)."
Normal search (direct similarity between query and document) only sees information returned from the wave source via a "single reflection."

However, truly important information (hubs) is strongly tied to many other relevant documents as well.
This reranker simulates waves bouncing multiple times (random walking) across the graph of search results, and by calculating the **"stationary probability reached after infinitely many multipath routes (Random Walk with Restart / Personalized PageRank),"** it inversely calculates (identifies) the true scattering source.

---

## Usage

It is used as a Post-processing step after performing a Top-K search in a Vector DB, etc.

```typescript
import { MultipathScatteringReranker } from 'warpvector';

// Initialize the reranker
const reranker = new MultipathScatteringReranker({
  alpha: 0.85,          // Attenuation rate of multipath scattering (inverse of teleport probability)
  threshold: 0.1,       // Similarity threshold to make the graph sparser
  maxIterations: 20,    // Maximum number of times to track multiple reflections
  tolerance: 1e-6       // Convergence criterion
});

// Group of top candidate vectors and initial scores obtained from a Vector DB, etc.
const candidates = [ /* ... Float32Array[] ... */ ];
const initialScores = [ 0.95, 0.82, 0.77 /* ... */ ];

// Execute reranking using multiple paths
const results = reranker.rerank(null, candidates, initialScores);

/*
results is sorted in descending order of score in the following format:
[
  { 
    originalIndex: 1, 
    score: 0.15,         // New score based on multipath scattering theory (Stationary probability)
    initialScore: 0.82,  // Initial cosine similarity
    vector: Float32Array 
  },
  ...
]
*/
```

---

## Parameter Tuning

### `alpha` (Multipath Scattering Attenuation Rate / Depth of Propagation)
- Default is `0.85`. This corresponds to the Damping Factor in the PageRank algorithm.
- **Increase (e.g., 0.9 ~ 0.95)**: Waves propagate far without attenuating. The structure of the entire network (massive hubs) is emphasized, and the influence of the direct similarity of the query (initial score) becomes relatively weaker.
- **Decrease (e.g., 0.1 ~ 0.5)**: Waves attenuate (absorb) quickly. This results in a mild correction closer to the initial cosine similarity, considering only the support of immediately neighboring documents.
- Setting it to `0.0` means no scattering at all, leaving the initial scores as they are.

### `threshold` (Edge Cutoff Threshold)
- Default is `0.0`.
- If the cosine similarity between candidates is less than this value, they are considered "unconnected (waves do not scatter)" on the graph.
- Setting it around `0.1` ~ `0.3` cuts off semantically weak noise scattering paths, enabling sharper aggregation.

---

## Differentiating from `TimeReversalReranker`

WarpVector includes another reranker using a physics analogy: `TimeReversalReranker`.

| Feature | TimeReversalReranker | MultipathScatteringReranker |
| :--- | :--- | :--- |
| **Physics Analogy** | Time-Reversal Mirror (Inverse Heat Equation / Wave Rewinding) | Multipath Scattering & Interference (Stationary field of Random Walk) |
| **Mathematical Model** | Forward Euler Method on Graph Laplacian | Random Walk with Restart on a Markov Chain |
| **Behavior** | Peaks "siphon up" surrounding scores | Finds hubs where scores "flow in" from many nodes |
| **Suitable Use Case** | "Sharpening" when similarities are blurry overall | When you want to drop isolated outliers and find "authoritative information" |

* Use `TimeReversalReranker` if you "just want to find the single sharpest item."
* Use `MultipathScatteringReranker` if you "want to find solid information supported by a lot of relevant information."
