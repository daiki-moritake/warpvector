# Troubleshooting Guide

A collection of common problems and solutions when using WarpVector.

---

## 1. WASM Initialization Error

### Symptom
```
WASM initialization failed, falling back to JS.
```

### Cause
- Running in an environment where WebAssembly is not supported.
- Restricted by the runtime's WASM execution policies (e.g., some Cloudflare Workers plans).

### Solution
1. **Automatic Fallback**: If WarpVector fails to initialize WASM, it automatically falls back to a pure JS implementation. This warning message is harmless, but performance will decrease.
2. **Explicit Initialization**: By calling `await pipeline.init()` at startup, you can detect WASM initialization failures early.
3. **Using autoInit**: As of v0.2.0, `WarpPipeline` defaults to `autoInit: true`. It automatically initializes WASM upon the first `runStream()` call.

---

## 2. Forgetting to call init()

### Symptom
- Results from `MlpAdapter` become zero vectors.
- WASM-related errors are thrown.

### Solution

**v0.2.0 and later (Recommended)**: Use the `autoInit` feature of `WarpPipeline` (enabled by default):

```typescript
const pipeline = new WarpPipeline(1536); // autoInit is true by default
// No need to call init(), it auto-initializes on the first runStream()
```

**When initializing explicitly**:
```typescript
const pipeline = new WarpPipeline(1536, { autoInit: false });
await pipeline.init(); // Call explicitly
```

---

## 3. Dimensionality Mismatch Error

### Symptom
```
IntentAdapter: Dimensionality of input vector does not match.
  Expected: 1536
  Actual: 768
```

### Cause
- The embedding model was changed, but the adapter's dimension setting wasn't updated.
- A pipeline step expects the pre-conversion dimensionality after a dimension transformation via `ProjectionAdapter`.

### Solution
1. **Check configuration with `pipeline.inspect()`**:
```typescript
console.log(pipeline.inspect());
// Pipeline [1536-dim]
//   Step 0: MlpAdapter
//   Step 1: IntentAdapter
//   Final: QuantizationAdapter
```

2. **Check intermediate outputs with `pipeline.dryRun()`**:
```typescript
const results = pipeline.dryRun(testVector, { intent: "tech" });
results.forEach(r => {
  console.log(`${r.step}: dim=${r.output.length}, ${r.durationMs.toFixed(2)}ms`);
});
```

3. **During Model Migration**: Use `MigrationTrainer` to learn a projection matrix that converts existing vectors into the new model's space.

---

## 4. Memory Limits in Edge Environments

### Symptom
- `WASM memory grow failed` error.
- The Worker crashes due to OOM (Out Of Memory).

### Cause
- Reached the Cloudflare Workers memory limit (e.g., 128MB).
- Attempting to process a massive number of vectors in a single batch.

### Solution

1. **Use Stream Processing**:
```typescript
// Process as a stream instead of a single batch (Memory efficient)
const results = pipeline.runStream(vectorGenerator, {
  batchSize: 64,  // Reduce the batch size
});
```

2. **Reduce Memory Using Quantization**:
```typescript
// Compress to 1/4 with Int8, or 1/32 with Binary from Float32
pipeline.setFinalStage("quantize", new QuantizationAdapter({
  type: "int8",  // or "binary"
  dim: 1536
}));
```

3. **Monitor WASM Memory Usage**:
```typescript
import { getWasmMemoryStats } from 'warpvector';

const stats = getWasmMemoryStats();
console.log(`Used: ${stats.usedBytes}, Peak: ${stats.peakBytes}, Total: ${stats.totalBytes}`);
```

---

## 5. Identifying Performance Bottlenecks

### Symptom
- Pipeline processing speed is slower than expected.
- Unsure which step is the bottleneck.

### Solution

**Enable Metrics Collection**:
```typescript
const pipeline = new WarpPipeline(1536)
  .addStep("mlp", mlpAdapter)
  .addIntent(intents);

// Enable metrics collection
pipeline.metrics.enable();

// Execute processing
for (const vec of vectors) {
  pipeline.run(vec, { intent: "tech" });
}

// Check results
const metrics = pipeline.metrics.getMetrics();
console.log(`Total runs: ${metrics.totalRuns}`);
console.log(`Avg run duration: ${metrics.avgRunDurationMs.toFixed(2)}ms`);
console.log("Step durations:");
for (const [step, avg] of Object.entries(metrics.avgStepDurationMs)) {
  console.log(`  ${step}: ${avg.toFixed(3)}ms`);
}
```

**Measure a single vector using `dryRun()`**:
```typescript
const results = pipeline.dryRun(testVector, { intent: "tech" });
results.forEach(r => {
  console.log(`${r.step}: ${r.durationMs.toFixed(3)}ms`);
});
```

---

## 6. Data Corruption in Concurrent Processing

### Symptom
- Processing multiple requests simultaneously results in corrupted vector outputs.

### Cause
- Because WASM memory is a shared resource, simultaneous access by multiple processes causes corruption.

### Solution
**Use `runStream`**: As of v0.2.0, `runStream` internally utilizes `wasmMutex` for exclusive control.

**When managing exclusive control manually**:
```typescript
import { wasmMutex } from 'warpvector';

// Execute processes utilizing WASM memory exclusively
const result = await wasmMutex.runExclusive(() => {
  return pipeline.run(vector, { intent: "tech" });
});
```

---

## 7. Errors during importState

### Symptom
```
WarpVector: Validation failed for field 'state'.
  Expected a JSON string, but received object.
```

### Cause
- You modified or parsed the result of `exportState()` before passing it to `importState()`.
- Data was corrupted during JSON serialization/deserialization.

### Solution
```typescript
// Correct usage
const state = adapter.exportState();    // string type
const json = JSON.stringify(state);     // E.g., when saving to Redis

// Upon restoration
const parsed = JSON.parse(json);        // Convert back to string
const restored = IntentAdapter.importState(parsed);
```
