# Dimensionality Reduction & Cross-Model Migration

## ProjectionAdapter — Dimensionality Reduction/Expansion via Projection Transformation

Transforms the dimensionality of a vector using projection matrices pre-calculated by PCA or SVD. Internally, it executes high-speed matrix-vector multiplications via WASM and features an automatic fallback to a pure JS implementation.

### Typical Use Cases

1. **Vector Compression**: Reduce 1536-dimensional embeddings to 512 or 256 dimensions to save memory and search costs.
2. **Feature Extraction**: Project into the principal component space obtained via PCA.
3. **Dimensionality Transformation within a Pipeline**: Can be integrated into a chain using `addProjection()` on a `WarpPipeline`.

### Usage

```typescript
import { ProjectionAdapter } from '@warpvector/core';

// Set a projection matrix from 1536 dimensions -> 512 dimensions
const adapter = new ProjectionAdapter(1536, 512, {
  pca: { matrix: pcaMatrix, bias: pcaBias }
});

// Execute dimensionality reduction (automatically uses WASM if available)
const compressed = adapter.tune(baseVector, "pca"); // 512-dimensional Float32Array
```

### Usage in a Pipeline

```typescript
import { WarpPipeline } from '@warpvector/core';

const pipeline = new WarpPipeline(1536)
  .addIntent(intentWeights)       // First, spatial deformation based on intent
  .addProjection(512, { v1: projWeights }); // Then, dimensionality reduction

// inputDim is automatically updated to 512 for subsequent steps
```

---

## AlignmentAdapter & MigrationTrainer — Zero-Downtime Model Migration

When switching embedding models (e.g., `text-embedding-ada-002` (1536D) → `text-embedding-3-small` (512D)), you usually have to re-index millions of vectors in your database, which costs time and money. 

The `AlignmentAdapter` completely solves this by translating new queries instantly into the old vector space using an affine transformation, eliminating the need for re-indexing (vendor lock-in broken).

### Workflow

1. Embed the same text (around 100~500 samples) using both the old and new models.
2. Add them as pair data to the `MigrationTrainer`.
3. Learn the projection matrix using the Adam optimizer.
4. Pass the learned `ProjectionWeights` to an `AlignmentAdapter` and execute.

### Usage

```typescript
import { MigrationTrainer } from "@warpvector/train";
import { AlignmentAdapter } from '@warpvector/core';

// We want to translate New Model (512D) queries to Old Model (1536D) space
const trainer = new MigrationTrainer(512, 1536);

// Add embedding pairs of the same text
trainer.addExample({ source: newSmallVec1, target: oldAdaVec1 });
trainer.addExample({ source: newSmallVec2, target: oldAdaVec2 });
trainer.addExample({ source: newSmallVec3, target: oldAdaVec3 });

// Learn (automatically search for the optimal learning rate with autoTune)
const alignmentWeights = await trainer.train({ epochs: 200, autoTune: true });

// Apply the learned projection matrix to an AlignmentAdapter
const migrator = new AlignmentAdapter(512, 1536, {
  migration: alignmentWeights,
});

// Convert the new vector into the old DB's space!
const translatedVector = migrator.align(newQueryVector, "migration");

// Now you can search your old database using `translatedVector`!
```

### autoTune (Optimal Learning Rate Search)

By specifying `train({ autoTune: true })`, it internally tests candidate learning rates for a short period and automatically selects the value that minimizes the loss.
