# Getting Started

`warpvector` is an ultra-lightweight middleware that enables dynamic transformation (affine transformations) of vector spaces based on "user intent" in vector searches.

## 1. Installation

Install using your preferred package manager.

```bash
# Using npm
npm install warpvector

# Using bun
bun add warpvector
```

No Python environment or heavy machine learning libraries (like PyTorch) are required. WebAssembly (WASM) is automatically loaded internally, allowing even large batch vector processing to run at high speed.

## 2. Basic Usage

The core of `warpvector` is the `IntentAdapter` class.

### Initializing the Adapter

First, prepare the transformation matrices and biases corresponding to each "intent" (context) and initialize the adapter. Here, we use the `WarpPipeline`, which is the simplest and most powerful way to assemble vector operations.

```typescript
import { WarpPipeline, IntentWeights } from 'warpvector';

const intents: Record<string, IntentWeights> = {
  // For example, a transformation definition for a "risk analysis" intent
  riskAnalysis: {
    matrix: [
      [1.2, 0.1, -0.4],
      [-0.1, 1.5, 0.2],
      [0.3, -0.2, 1.1],
    ],
    bias: [0.05, -0.1, 0.2]
  }
};

// Build a pipeline that takes a 3D vector, performs Intent transformation, and quantizes it
const pipeline = new WarpPipeline(3)
  .addIntent(intents)
  .quantize("int8");
```

### Transforming (Warping) Vectors

Transform the standard embedding vectors output by an AI model according to the specified intent.

```typescript
// Original vector output by the AI model
const baseVector = [0.15, -0.23, 0.88];

// Transform the vector according to the "riskAnalysis" intent and quantize to Int8
const tunedVector = pipeline.run(baseVector, { intent: "riskAnalysis" });

console.log(tunedVector); // The new vector converted to Int8Array
```

## 3. Normalization and Similarity Calculation

Transformed vectors can be directly passed to vector databases like Pinecone or Qdrant. However, if you are comparing them using cosine similarity, it is useful to normalize them (setting the L2 Norm to 1) after transformation.

```typescript
import { normalize, cosineSimilarity } from 'warpvector';

const normalized = normalize(tunedVector);

// Calculate similarity with another vector
const queryVector = normalize([0.1, 0.0, 0.9]);
const similarity = cosineSimilarity(normalized, queryVector);
console.log(`Similarity: ${similarity}`);
```

For more advanced usage (automatic blending, WASM batch processing, etc.), please refer to [Advanced Usage](./advanced-usage.md).

---

## 4. Running the Quickstart Sample

We provide a sample script where you can actually see the effects of `warpvector` in your terminal. This script lets you experience the difference in scores between a "normal search" and a search "warped (affine transformed) towards technology" against three predefined document vectors.

With the repository cloned, run the following command:

```bash
# If using Bun
bun run examples/quickstart.ts

# If using Node.js (requires ts-node, etc.)
npx ts-node examples/quickstart.ts
```

### Overview of the Sample Code

The following processes take place in `examples/quickstart.ts`:
1. **Defining Documents**: Prepares three sample vectors representing "Weather", "Economy", and "Technology".
2. **Normal Search**: The closest document to the query is "Economy".
3. **Applying Warp**: Uses `IntentAdapter` to apply an affine transformation that expands (scales up) the "Z-axis (technology feature)" of the vector space by 2.5x, and shifts the entire space along the Z-axis.
4. **Search Reflecting Intent**: Even though the vectors on the database side (document side) are not modified at all, the search query is warped into a "technology-focused space", causing the technology document to leap to the top of the search score rankings.

In this way, simply by inserting `warpvector` at the frontend (or edge layer), you can instantly achieve ranking manipulation of search results tailored to the user's intent in real-time!
