# 14. Soft Whitening

`SoftWhiteningAdapter` is a powerful component that resolves the "bias towards general contexts (Semantic Diffusion)" peculiar to embedding vectors of Large Language Models (LLMs) by using **Principal Component Attenuation**.

## Why is "Soft Whitening" needed?

General-purpose models like OpenAI's `text-embedding-ada-002` and `text-embedding-3-small` excel at capturing the broad context of text. However, when searching within specific domains (medical, legal, internal company jargon, etc.), the following issues occur:

*   **Excessive Context Mixing:** Even if you search using a niche keyword, it gets pulled toward similar common words, causing the vector to "blur."
*   **The "Everything is Similar" Problem:** Principal components with large variances (frequent words or general stylistic components) dominate the similarity calculation, burying the sharp intent (Sharp Source) you originally wanted to find.

To solve this, **Soft Whitening** highlights the specific niche meanings by applying a soft attenuation (filtering) to the principal components with large variances (which are spread out across general contexts).

---

## How it works

`SoftWhiteningAdapter` tracks the "Top $K$ Principal Components" and their "variances (eigenvalues $\lambda_k$)" online from streaming data.

Then, during inference, it uses a sharpness parameter $\tau$ (tau) to strongly attenuate (filter) the components with larger variances.
Mathematically, it applies the following exponential decay to the projected components in the direction of the principal components:

$$ x'_{proj} = e^{-\tau \lambda_k} x_{proj} $$

(In the internal implementation, it subtracts the following attenuation coefficient)
$$ \text{Attenuation}(k) = 1 - \exp(-\tau \lambda_k) $$

- Directions with a large eigenvalue $\lambda_k$ (diffused in common, general contexts) are heavily reduced.
- Directions with a small eigenvalue (possessing unique, sharp meanings) are mostly preserved.

---

## Implementation Example

```typescript
import { SoftWhiteningAdapter } from 'warpvector';

// Vector dimension 1536 (OpenAI ada-002, etc.)
// An adapter that tracks the variance of the eigenspace and applies a soft whitening filter
const adapter = new SoftWhiteningAdapter(1536, {
  learningRate: 0.01,
  numComponents: 5,   // Track the top 5 components
  tau: 2.0,           // Sharpness strength. Larger values suppress more strongly.
  normalizeOutput: true // L2 normalize the output (true if Cosine Similarity is assumed)
});

// --- 1. Online Tracking ---
// Learn how the space spreads out from user queries or corpus documents, etc.
adapter.update(vectorA);
adapter.update(vectorB);
adapter.update(vectorC);

// --- 2. Sharpening (Inference) ---
// Resolve contextual muddiness during search and transform it into the original sharp semantic space.
const sharpVector = adapter.tune(queryVector);

// --- 3. Batch Processing (Optimization) ---
// Fast batch processing is possible for multiple vectors (e.g., when creating a DB index) using `tuneBatch`.
const sharpVectors = adapter.tuneBatch([vector1, vector2, vector3]);
```

---

## Parameter Tuning Guide

This explains the parameters that can be specified in `SoftWhiteningConfig` along with best practices.

### 1. `tau` (Sharpness Strength)
The most important parameter. The default is `1.0`.
- `tau = 0`: No correction is performed (disabled).
- `tau = 0.5 ~ 2.0`: Mild sharpening. Recommended when you want to reduce contextual noise in general RAG (Retrieval-Augmented Generation).
- `tau = 5.0 ~`: Extremely strong sharpening. The top principal components are almost completely removed.
- `tau → ∞`: Asymptotically approaches the behavior of `WhiteningAdapter` (complete orthogonalization).

### 2. `numComponents` (Number of Principal Components to Track)
Usually set between `1` and `10`. The default is `5`.
If set too large, it may shave off important niche components that have meaning, so adjust it according to the diversity of your corpus.

### 3. `normalizeOutput` (Output Normalization)
The default is `true`.
Because the length in the component directions is reduced, the vector's overall norm (length) deviates from 1.0.
If you are assuming **Cosine Similarity** (not dot product) in Pinecone, Qdrant, pgvector, etc., you must set this to `true` to return the length back to 1.0.

---

## Differences from `WhiteningAdapter`

`WarpVector` also includes a `WhiteningAdapter` to remove spatial bias.
They have similar goals, but are used differently as follows:

| Feature | WhiteningAdapter | SoftWhiteningAdapter |
| :--- | :--- | :--- |
| **Approach** | Completely "removes" anisotropy (bias) | "Smoothly attenuates" based on the variance magnitude |
| **Information Loss** | Information of top components is completely lost | Top components also remain slightly, making it less prone to breaking |
| **Parameters** | Only `numComponents` | `numComponents` and `tau` (Smoothness can be adjusted) |
| **Suitable Use Case** | When you want to eliminate the strong fixed bias inherent to language models | When you want to sharpen search queries that have mixed contexts |

For products requiring more advanced and delicate tuning (adjustments that do not overly destroy the search intent), we strongly recommend adopting the `SoftWhiteningAdapter`.
