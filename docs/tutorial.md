# WarpVector E2E Tutorial: Creating an Intent Matrix from Scratch

In this tutorial, we will build a practical scenario from start to finish using WarpVector: **"Dynamically transforming a search query's vector to match a specific intent."**

## Table of Contents

1. [Concept: What is an Intent Matrix?](#concept-what-is-an-intent-matrix)
2. [Setup](#setup)
3. [Step 1: Building a Linear Transformation with IntentAdapter](#step-1-building-a-linear-transformation-with-intentadapter)
4. [Step 2: Assembling the Pipeline](#step-2-assembling-the-pipeline)
5. [Step 3: Reducing Storage Costs with Quantization](#step-3-reducing-storage-costs-with-quantization)
6. [Step 4: Saving and Restoring State](#step-4-saving-and-restoring-state)
7. [Step 5: Non-linear Transformation with MLP Adapter](#step-5-non-linear-transformation-with-mlp-adapter)
8. [API Reference](#api-reference)

---

## Concept: What is an Intent Matrix?

In vector search, the optimal vector space differs even for the same query depending on whether the focus is on **"similarity"** or **"diversity"**. WarpVector's **Intent Matrix** achieves this "rotation and scaling of the vector space according to intent."

```
Original Vector Space      Intent Matrix W        Transformed Space
     [v₁]               × [W_intent]   →    [v₁']
     [v₂]                                   [v₂']
```

Mathematically, it is an affine transformation: `v' = Wv + b`.

---

## Setup

```bash
# Create a new project
mkdir my-warp-project && cd my-warp-project
bun init -y

# Install WarpVector
bun add @warpvector/core
```

---

## Step 1: Building a Linear Transformation with IntentAdapter

The most basic usage is to register an intent matrix with the `IntentAdapter` and transform the vector.

```typescript
import { IntentAdapter } from "@warpvector/core";

// Create an IntentAdapter that handles 3-dimensional vectors
const adapter = new IntentAdapter(3);

// Register an intent matrix focused on "similarity"
// This matrix represents a transformation that doubles the 1st dimension (semantic similarity),
// and halves the 3rd dimension (diversity score).
adapter.addIntent("similarity", {
  matrix: [
    [2, 0, 0],   // Double the 1st dimension
    [0, 1, 0],   // Keep the 2nd dimension as is
    [0, 0, 0.5], // Halve the 3rd dimension
  ],
  bias: [0, 0, 0], // No bias
});

// An intent matrix focused on "diversity"
adapter.addIntent("diversity", {
  matrix: [
    [0.5, 0, 0], // Halve the 1st dimension
    [0, 1, 0],
    [0, 0, 2],   // Double the 3rd dimension
  ],
  bias: [0, 0, 0],
});

// Transform the vector
const query = [0.8, 0.5, 0.3];
const forSimilarity = adapter.tune(query, "similarity");
// → Float32Array [1.6, 0.5, 0.15]

const forDiversity = adapter.tune(query, "diversity");
// → Float32Array [0.4, 0.5, 0.6]
```

---

## Step 2: Assembling the Pipeline

You can chain multiple transformations serially as a **Pipeline**.

```typescript
import { WarpPipeline } from "@warpvector/core";

// Pipeline: Intent Transformation → Dimensionality Reduction (3D → 2D)
const pipeline = new WarpPipeline(3)
  .addIntent({
    search: {
      matrix: [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      bias: [0.1, 0, 0],
    },
  })
  .addProjection(2, {
    search: {
      matrix: [
        [1, 0, 0], // Extract only the 1st and 2nd dimensions
        [0, 1, 0],
      ],
    },
  });

// Execute
const result = pipeline.run([0.5, 0.8, 0.3], { intent: "search" });
// → Float32Array [1.1, 0.8]  (Compressed from 3D → 2D)
```

---

## Step 3: Reducing Storage Costs with Quantization

Using the `QuantizationAdapter` from `@warpvector/extras`, you can quantize Float32 vectors into Int8 to reduce storage costs by **75% (1/4)**.

```typescript
import { WarpPipeline } from "@warpvector/core";
import { QuantizationAdapter } from "@warpvector/extras";

const quantizer = new QuantizationAdapter({ type: "int8", dim: 2 });

// Set quantization at the final stage of the pipeline (FinalStageAdapter pattern)
const pipeline = new WarpPipeline(3)
  .addIntent({
    search: {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      bias: [0, 0, 0],
    },
  })
  .addProjection(2, {
    search: {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
      ],
    },
  })
  .setFinalStage("QuantizationAdapter", quantizer);

const result = pipeline.run([0.5, 0.8, 0.3], { intent: "search" });
// → Int8Array [64, 102]  (Quantized from Float32Array → Int8Array)
```

> **Note:** `setFinalStage()` differs from the traditional `addStep("QuantizationAdapter", ...)`. It is type-safe by enforcing that the intermediate stages of the pipeline remain exclusively Float32Array.

---

## Step 4: Saving and Restoring State

All pipeline weights (intent matrices, projection matrices, etc.) can be serialized to and restored from JSON.

```typescript
// Save
const state = pipeline.exportState();
const json = JSON.stringify(state);
// → Save to a database or file

// Restore
const restored = WarpPipeline.importState(JSON.parse(json));
const result2 = restored.run([0.5, 0.8, 0.3], { intent: "search" });
// → Produces the exact same result as before saving
```

---

## Step 5: Non-linear Transformation with MLP Adapter

For complex spatial transformations that cannot be expressed with linear transformations, use `MlpAdapter` (Multi-Layer Perceptron). With fast inference powered by WASM, even 1536-dimensional vectors can be processed in a few microseconds.

```typescript
import { MlpAdapter } from "@warpvector/ml";

const mlp = new MlpAdapter([
  {
    // Layer 1: 3 → 4 (ReLU)
    matrix: [
      [0.5, 0.3, 0.1],
      [0.2, 0.8, 0.4],
      [0.1, 0.1, 0.9],
      [0.4, 0.2, 0.3],
    ],
    bias: [0.1, 0, 0, 0.05],
    activation: "relu",
  },
  {
    // Layer 2: 4 → 2 (Linear)
    matrix: [
      [1, 0, 0.5, 0.3],
      [0, 1, 0.2, 0.8],
    ],
    bias: [0, 0],
    activation: "linear",
  },
]);

// Initialize WASM (only once)
await mlp.init();

// Infer
const output = mlp.tune([0.5, 0.8, 0.3]);
// → Float32Array (Result of non-linear transformation reduced to 2 dimensions)

// Integrate into the pipeline
const pipeline = new WarpPipeline(3)
  .addStep("MlpAdapter", mlp);

await pipeline.init(); // Initialize WASM adapters
const pipelineResult = pipeline.run([0.5, 0.8, 0.3]);
```

---

## API Reference

| Class | Use Case | Package |
|--------|------|-----------|
| `IntentAdapter` | Linear transformation per intent (Wx+b) | `@warpvector/core` |
| `LoraIntentAdapter` | Low-rank intent transformation using LoRA | `@warpvector/core` |
| `ProjectionAdapter` | Dimensionality reduction via PCA/SVD | `@warpvector/core` |
| `WarpPipeline` | Serial connection of adapters | `@warpvector/core` |
| `QuantizationAdapter` | Int8/Binary quantization | `@warpvector/extras` |
| `ColbertAdapter` | Late Interaction (MaxSim) | `@warpvector/extras` |
| `MlpAdapter` | Non-linear transformation (WASM MLP) | `@warpvector/ml` |
| `WhiteningAdapter` | Whitening normalization | `@warpvector/ml` |

### Pipeline Design Pattern

```
Input Vector (Float32Array)
    │
    ├─ WarpAdapter (IntentAdapter, ProjectionAdapter, MlpAdapter...)
    │   → Always returns a Float32Array
    │   → Can be connected serially
    │
    └─ FinalStageAdapter (QuantizationAdapter)
        → Returns Int8Array / Uint8Array
        → Can only be placed at the final stage of the pipeline
        → Configured using setFinalStage()
```
