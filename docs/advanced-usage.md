# Advanced Usage

This section introduces more advanced use cases of `warpvector` and features designed to maximize performance.

---

## 1. Non-linear Activations

Because simple affine transformations (linear) cannot fully separate complex concepts, functions like `tune` allow you to optionally specify a **non-linear activation function**.

```typescript
// Truncate negative values to 0 (ReLU)
const reluVector = adapter.tune(baseVector, "riskAnalysis", "relu");

// Compress the space into the 0.0 ~ 1.0 range (Sigmoid)
const sigmoidVector = adapter.tune(baseVector, "riskAnalysis", "sigmoid");

// Compress the space into the -1.0 ~ 1.0 range (Tanh)
const tanhVector = adapter.tune(baseVector, "riskAnalysis", "tanh");
```

---

## 2. Auto-blending

Even if the user doesn't explicitly state "I want to search with 70% risk analysis and 30% economic impact," the system can automatically infer and synthesize the most appropriate intent ratio directly from the meaning of the input query vector itself.

To use this, you set a `routingVector` (a representative vector for that intent) when initializing each intent.

```typescript
const adapter = new IntentAdapter({
  intentA: {
    matrix: [...], bias: [...],
    routingVector: [1.0, 0.0, 0.0] // Representative vector for intentA
  },
  intentB: {
    matrix: [...], bias: [...],
    routingVector: [0.0, 1.0, 0.0] // Representative vector for intentB
  }
});

// Automatically calculates the ratio of intentA and intentB using Cosine Similarity + Softmax 
// based on the query vector's content, and applies the blended transformation
const autoTuned = adapter.tuneAutoBlended(queryVector);
```

---

## 3. Ultra-fast Batch Processing with WASM / SIMD

In use cases where you retrieve 10,000 records from a vector database like Pinecone and re-rank them on the frontend or edge server, single transformations using a `for` loop can become a bottleneck.

In `warpvector`, calling `tuneBatch` or `tuneBatchBlended` automatically invokes the WebAssembly (WASM) module internally.

```typescript
// Array of vectors (batch)
const batchVectors = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  // ... 10,000 records
];

// Internally transferred in bulk to WASM shared memory, executing optimized Float32 calculations
const tunedBatch = adapter.tuneBatch(batchVectors, "riskAnalysis");
```
You don't need to worry about loading WASM or managing memory on the user side. It works seamlessly in the browser, Node.js, and Cloudflare Workers.

---

## 4. Optimizing Ultra-High Dimensional Vectors with LoRA Adapters

For ultra-high dimensional vectors like OpenAI's `text-embedding-3-small` (1536 dimensions), a normal `IntentAdapter` requires a full matrix of `1536 x 1536` (about 2.36 million parameters), increasing memory and computational complexity.

`LoraIntentAdapter` decompresses this full matrix into "low-rank" matrices A and B (e.g., rank 16), maintaining expressive power while compressing the parameter count to 1536 * 16 * 2 = 49,152 (about a 98% reduction).

```typescript
import { LoraIntentAdapter } from 'warpvector';

// Initialize with 1536 dimensions, rank 16
const loraAdapter = new LoraIntentAdapter(1536, 16);

loraAdapter.addIntent("myContext", {
  matrixA: [...], // 1536 x 16
  matrixB: [...], // 16 x 1536
  bias: [...]     // 1536
});

const tuned = loraAdapter.tune(baseVector, "myContext");
```
This applies the techniques used in Large Language Model (LLM) fine-tuning directly to in-memory vector search spaces.

---

## 5. Auto-training Intent Matrices with IntentTrainer

Manually defining massive transformation matrices is difficult. By using `IntentTrainer`, you can **automatically learn the optimal intent matrix (IntentWeights)** from a small amount of sample data representing "for this input vector, I want this result vector returned."

```typescript
import { IntentTrainer } from 'warpvector';

// 1. Initialize the trainer by specifying the dimensionality
const trainer = new IntentTrainer(1536);

// 2. Add training data (positive examples)
trainer.addExample({
  input: [...],  // Search query vector
  target: [...]  // Ideal document vector
});

// 3. Learn the optimal matrix (W) and bias (b) using Stochastic Gradient Descent (SGD)
const learnedWeights = trainer.train({
  learningRate: 0.05,
  epochs: 200,
  regularization: 0.001
});

// 4. Integrate the learned matrix into the adapter
adapter.addIntent("user_personalized_intent", learnedWeights);
```

### Online Learning (Feedback Loop)
It is also possible to fine-tune (learn) the matrix in real-time every time a user clicks on a search result.

```typescript
// The vector of the result the user clicked (Ideal vector)
const clickedVector = [...];

// Update (fine-tune) the current intent matrix by just one step
const updatedWeights = trainer.updateOnline(
  currentWeights, // Current IntentWeights
  queryVector,    // Input query
  clickedVector,  // Clicked target
  0.01            // Learning rate (set low)
);

// Apply the updated intent
adapter.addIntent("user_personalized_intent", updatedWeights);
```
With this, the more the system is used, the smarter the search space becomes, closely aligning with individual intents (contexts).

---

## 6. Binary Serialization (Ultra-lightweight Save and Restore)

Saving a massive learned intent matrix as JSON results in huge file sizes, consuming massive memory and time during parsing (`JSON.parse`). `warpvector` supports ultra-fast serialization using a highly lightweight **binary format (Uint8Array)**.

### Export (Save)
```typescript
// Extract a learned or predefined intent as binary
const binaryData: Uint8Array = adapter.exportIntentBinary("user_personalized_intent");

// (For Node.js/Bun environments, save to the file system)
import * as fs from "fs";
fs.writeFileSync("user_intent.wrpv", binaryData);
```

### Import (Restore)
```typescript
import { IntentAdapter } from "warpvector";

// (For Node.js/Bun environments, read from file)
import * as fs from "fs";
const loadedBinary = fs.readFileSync("user_intent.wrpv");

// Create an empty adapter specifying only the dimensionality
const adapter = new IntentAdapter(1536);

// Load the binary data and register it as a new intent (Ultra-fast, no JSON parsing required)
adapter.importIntentBinary("restored_intent", loadedBinary);

// Ready for inference immediately
const result = adapter.tune(queryVector, "restored_intent");
```
This feature is immensely powerful when you want to load a user's personalized data at high speed in edge environments (like Cloudflare Workers).

---

## 7. Migration Between Different Embedding Models via MigrationTrainer

In AI development, updating an embedding model (e.g., migrating from `text-embedding-ada-002` to `text-embedding-3-small`) requires resending tens of millions of records to the API, creating a massive barrier in cost and time.

`MigrationTrainer` automatically learns a **translation matrix (projection) from the "old model's space" to the "new model's space"**. By simply preparing a small number of pair data (anchors) that vectorize common text with both models, you can warp the vectors in your existing database into the "new model's space" in memory.

### Learning the Migration Matrix
```typescript
import { MigrationTrainer, ProjectionAdapter } from "warpvector";

// Migration from the Old Model (e.g., 1536 dim) to the New Model (e.g., 512 dim)
const trainer = new MigrationTrainer(1536, 512);

// Add a small number of anchor vector pairs
trainer.addExample({
  source: [...], // Vector encoded with the old model (ada-002)
  target: [...]  // Vector encoded with the new model (3-small)
});

// Learn the translation matrix using Momentum-SGD
const learnedWeights = trainer.train({
  learningRate: 0.05,
  epochs: 300
});

// ---------------------------------------------------------
// Set the learned weights into a ProjectionAdapter and use
// ---------------------------------------------------------
const adapter = new ProjectionAdapter(1536, 512);
adapter.addProjection("migrate_to_v3", learnedWeights);

// Warp-transform vectors fetched from the old database into the new model's space!
const oldVector = db.get("document_id"); 
const newVector = adapter.project(oldVector, "migrate_to_v3");
```

This feature allows you to instantly integrate the search accuracy and characteristics of a new model into your existing system without incurring API costs.
