# Migration Guide

This guide covers breaking changes and migration steps for each major version upgrade.

- [v0.3 → v0.4](#v03--v04) (latest)
- [v0.2 → v0.3](#v02--v03)
- [v0.1 → v0.2](#v01--v02)

---

## v0.3 → v0.4

### Breaking Changes

#### 1. QuantizationAdapter API Change

`tune()` method has been removed. Use `encode()` for all quantization operations.

```diff
- const quantized = quantizer.tune(vector);
+ const quantized = quantizer.encode(vector);
```

#### 2. Training Utilities Moved to `@warpvector/train`

`SoftWhiteningAdapter` and related training modules have been moved from `@warpvector/ml` to the new `@warpvector/train` package.

```diff
- import { SoftWhiteningAdapter } from 'warpvector/ml';
+ import { SoftWhiteningAdapter } from 'warpvector/train';
```

```bash
# If using sub-packages directly:
npm install @warpvector/train
```

#### 3. Adam Optimizer Removed from ML Package

The built-in Adam optimizer has been removed from `@warpvector/ml`. Use `@warpvector/train` instead for training-related tasks.

### New Packages in v0.4

| Package | Purpose |
|---------|---------|
| `@warpvector/train` | Training, fine-tuning, and auto-ML tools |
| `@warpvector/rerank` | ColBERT and advanced graph-based rerankers |
| `@warpvector/eval` | RAG evaluation kit (Precision@K, Recall@K, NDCG, MRR, MAP) |

### Migration Steps

1. **Update dependency**: `npm install warpvector@latest`
2. **Replace `tune()` calls**: Change `quantizer.tune(v)` → `quantizer.encode(v)`
3. **Update training imports**: Move `SoftWhiteningAdapter` imports from `warpvector/ml` to `warpvector/train`
4. **Install new packages** (if using sub-packages): `npm install @warpvector/train @warpvector/rerank @warpvector/eval`

---

## v0.2 → v0.3

### Breaking Changes

> **There are no breaking changes in v0.3.** Existing v0.2 code will continue to work.

### New Features

- **WarpTracer**: Zero-dependency OpenTelemetry-compatible tracing
- **Cloudflare Vectorize** integration in `VectorDBAdapter`
- **IntentMatrixFactory**: Auto-learn intent matrices from category samples
- **`@warpvector/experimental`** package for unstable features

### Migration Steps

1. **Update dependency**: `npm install warpvector@latest`
2. **Optional**: Add `WarpTracer` for production monitoring
3. **Optional**: Use `IntentMatrixFactory` to replace hand-crafted intent matrices

---

## v0.1 → v0.2

### Breaking Changes

> **There are no breaking changes in v0.2.** Existing code will continue to work as is.

However, existing error handling code might be affected by the following changes:

#### Change in Validation Error Types

Errors thrown by validation functions (e.g., `assertType`, `assertArray`) have been changed from `Error` to `WarpValidationError`.

```diff
- catch (e) {
-   // e.message: "Invalid state: field 'matrix' must be an array"
- }
+ catch (e) {
+   if (e instanceof WarpValidationError) {
+     console.error(`${e.component}: ${e.field} - ${e.message}`);
+   }
+ }
```

> Because `WarpValidationError` inherits from `Error`, you can still catch it using `catch (e: Error)` as before.

---

### Utilizing New Features

#### 1. Automatic Pipeline Initialization (`autoInit`)

**v0.1 and earlier:**
```typescript
const pipeline = new WarpPipeline(1536).addStep("mlp", mlpAdapter);
await pipeline.init(); // Causes bugs if forgotten
pipeline.run(vector);
```

**v0.2 and later:**
```typescript
const pipeline = new WarpPipeline(1536).addStep("mlp", mlpAdapter);
// OK to omit init() — automatically initializes on the first runStream()
for await (const result of pipeline.runStream(vectors)) { /* ... */ }
```

> `autoInit` is enabled by default. If you wish to disable it, use `new WarpPipeline(1536, { autoInit: false })`.

#### 2. Improved Debugging with Structured Errors

**v0.1 and earlier:**
```typescript
try {
  pipeline.run(vector);
} catch (e) {
  console.error(e.message); // Generic error message
}
```

**v0.2 and later:**
```typescript
import { WarpPipelineError, WarpValidationError } from 'warpvector';

try {
  pipeline.run(vector);
} catch (e) {
  if (e instanceof WarpPipelineError) {
    console.error(`Step ${e.stepIndex} (${e.stepType}): ${e.message}`);
    console.error("Original cause:", e.cause);
  }
}
```

#### 3. Metrics Collection

```typescript
pipeline.metrics.enable();
pipeline.run(vector);

const stats = pipeline.metrics.getMetrics();
console.log(`Avg: ${stats.avgRunDurationMs.toFixed(2)}ms`);
console.log("Per step:", stats.avgStepDurationMs);
```

#### 4. Debugging Assistance (`inspect` / `dryRun`)

```typescript
// Check pipeline configuration
console.log(pipeline.inspect());

// Check intermediate outputs of each step
const debug = pipeline.dryRun(testVector, { intent: "tech" });
debug.forEach(r => console.log(`${r.step}: dim=${r.output.length}`));
```

#### 5. WASM Memory Monitoring

```typescript
import { getWasmMemoryStats } from 'warpvector';

const stats = getWasmMemoryStats();
console.log(`Peak memory: ${(stats.peakBytes / 1024).toFixed(0)}KB`);
```

---

## List of Error Classes

| Class | Purpose |
|--------|------|
| `WarpError` | Base class for all WarpVector errors |
| `WarpPipelineError` | Failure at a specific pipeline step (includes `stepIndex`, `stepType`) |
| `WarpDimensionMismatchError` | Dimension mismatch (includes `expectedDim`, `actualDim`) |
| `WarpInitializationError` | Method call before initialization completes |
| `WarpValidationError` | Validation failure in importState/configuration (includes `component`, `field`) |
