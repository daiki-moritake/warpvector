# Dimensionality Reduction & Cross-Model Migration (Projection & Migration)

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

### State Persistence

```typescript
// Export
const state = adapter.exportState(); // JSON string

// Import
const restored = ProjectionAdapter.importState(state);
```

---

## MigrationTrainer — Translating Vector Spaces Between Models

When switching embedding models (e.g., `text-embedding-ada-002` → `text-embedding-3-small`), this automatically learns a projection matrix to translate existing vectors into the space of the new model.

This allows you to approximately map old vectors into the new space without needing to re-embed all your data with the new model.

### Workflow

1. Embed the same text using both the old and new models.
2. Add them as pair data to the `MigrationTrainer`.
3. Learn the projection matrix using the Adam optimizer.
4. Pass the learned `ProjectionWeights` to a `ProjectionAdapter` and execute.

### Usage

```typescript
import { MigrationTrainer } from '@warpvector/ml';
import { ProjectionAdapter } from '@warpvector/core';

// Learn a projection matrix from 1536 dimensions (ada-002) -> 512 dimensions (3-small)
const trainer = new MigrationTrainer(1536, 512);

// Add embedding pairs of the same text
trainer.addExample({ source: adaVec1, target: smallVec1 });
trainer.addExample({ source: adaVec2, target: smallVec2 });
trainer.addExample({ source: adaVec3, target: smallVec3 });

// Learn (automatically search for the optimal learning rate with autoTune)
const projWeights = await trainer.train({ epochs: 200, autoTune: true });

// Apply the learned projection matrix to a ProjectionAdapter
const migrator = new ProjectionAdapter(1536, 512, {
  migration: projWeights,
});

// Convert the old vector into the new space
const newSpaceVector = migrator.tune(oldAdaVector, "migration");
```

### autoTune (Optimal Learning Rate Search)

By specifying `train({ autoTune: true })`, it internally tests 5 candidate learning rates (0.1, 0.05, 0.01, 0.005, 0.001) for a short period and automatically selects the value that minimizes the loss.
