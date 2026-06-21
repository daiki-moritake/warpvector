# State Persistence & Serialization (Universal Serialization)

Many of WarpVector's adapters (especially PCA learning results from `WhiteningAdapter`, and optimized weights from `IntentAdapter` / `MlpAdapter` via `Trainer`) hold "state" at runtime.

Serverless/Edge environments like Cloudflare Workers and Vercel Edge Functions are volatile (stateless), meaning that once a request finishes, any state in memory disappears.
Therefore, WarpVector provides a **Universal Serialization feature that can safely and instantly serialize/deserialize the state of all adapters in JSON format**.

## Supported Adapters

Currently, `exportState()` and the static method `importState()` are supported in all of the following adapters:

- `IntentAdapter` / `LoraIntentAdapter`
- `WhiteningAdapter`
- `ProjectionAdapter`
- `MlpAdapter`

## Basic Usage

### Saving (Export)

When streaming learning or data updates are finished, call `exportState()` and save the resulting string (JSON format) to Redis, Cloudflare KV, S3, or a text column in an RDB.

```typescript
import { WhiteningAdapter } from 'warpvector';

const adapter = new WhiteningAdapter(1536);

// ... Receive vectors and execute online learning (update) multiple times ...
adapter.update(vectorA);
adapter.update(vectorB);

// Serialize the state after learning (principal component vectors, etc.)
const stateString = adapter.exportState(); 
// -> '{"dimension":1536,"learningRate":0.001, ... "components":[...]}'

// Example: Save to Redis, etc.
await redis.set("my_whitening_state", stateString);
```

### Restoring (Import)

Upon the next startup or when processing on a different instance, simply pass the saved string to `importState()` (the class's static method), and an adapter in the exact same state will be instantly restored.

```typescript
import { WhiteningAdapter } from 'warpvector';

// Example: Load state from Redis, etc.
const stateString = await redis.get("my_whitening_state");

// Restore the instance
const restoredAdapter = WhiteningAdapter.importState(stateString);

// You can immediately perform search tuning or further learning, inheriting the previous learning results
const whitenedVector = restoredAdapter.tune(newVector);
```

## Important Notes on Restoring MlpAdapter

Because `MlpAdapter` uses WebAssembly (WASM) on the backend, you must reconstruct the WASM memory and instance after restoration.
Therefore, immediately after restoring with `MlpAdapter.importState()`, you **must always call `await mlp.init()`** (failing to do so will result in an error).

```typescript
import { MlpAdapter } from 'warpvector';

const stateString = await redis.get("my_mlp_state");

// 1. Restore the state as a JS object
const mlp = MlpAdapter.importState(stateString);

// 2. Reinitialize the WASM instance (async)
await mlp.init();

// 3. Ready for inference
const result = mlp.tune(vector);
```

## About Binary Serialization

In addition to JSON serialization, `IntentAdapter` also features `exportIntentBinary()` / `importIntentBinary()` methods that directly output/input a `Uint8Array`.
In environments requiring extreme performance where you want to eliminate even the overhead of JSON parsing (e.g., saving to IndexedDB or before mapping to WebGL textures), you can use this binary serialization instead.
