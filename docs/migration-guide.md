# Migration Guide (v0.1 → v0.2)

In WarpVector v0.2, we have significantly enhanced scalability and the Developer Experience (DX).
While backward compatibility with existing code is maintained, the migration steps to utilize new features are summarized below.

---

## Breaking Changes

> **There are no breaking changes in v0.2.** Existing code will continue to work as is.

However, existing error handling code might be affected by the following changes:

### Change in Validation Error Types

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

## Utilizing New Features

### 1. Automatic Pipeline Initialization (`autoInit`)

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

### 2. Improved Debugging with Structured Errors

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

### 3. Metrics Collection

```typescript
pipeline.metrics.enable();
pipeline.run(vector);

const stats = pipeline.metrics.getMetrics();
console.log(`Avg: ${stats.avgRunDurationMs.toFixed(2)}ms`);
console.log("Per step:", stats.avgStepDurationMs);
```

### 4. Debugging Assistance (`inspect` / `dryRun`)

```typescript
// Check pipeline configuration
console.log(pipeline.inspect());

// Check intermediate outputs of each step
const debug = pipeline.dryRun(testVector, { intent: "tech" });
debug.forEach(r => console.log(`${r.step}: dim=${r.output.length}`));
```

### 5. WASM Memory Monitoring

```typescript
import { getWasmMemoryStats } from 'warpvector';

const stats = getWasmMemoryStats();
console.log(`Peak memory: ${(stats.peakBytes / 1024).toFixed(0)}KB`);
```

---

## List of New Error Classes

| Class | Purpose |
|--------|------|
| `WarpError` | Base class for all WarpVector errors |
| `WarpPipelineError` | Failure at a specific pipeline step (includes `stepIndex`, `stepType`) |
| `WarpDimensionMismatchError` | Dimension mismatch (includes `expectedDim`, `actualDim`) |
| `WarpInitializationError` | Method call before initialization completes |
| `WarpValidationError` | Validation failure in importState/configuration (includes `component`, `field`) |

---

## Recommended Migration Steps

1. **Update Dependency**: `npm install warpvector@latest`
2. **Review `init()` Calls**: Can be removed if unnecessary due to `autoInit`
3. **Enhance Error Handling**: Utilize `WarpPipelineError` to identify the failing step
4. **Enable Metrics**: Add `pipeline.metrics.enable()` in development environments to identify bottlenecks
5. **Utilize `inspect()`**: Use debug outputs to verify pipeline configurations
