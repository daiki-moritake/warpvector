# Auto-Learning Implementation Guide

The most prominent feature of Warpvector is its ability to integrate an "auto-optimizing learning loop" based on user search and click behavior.
This document explains how to implement an **end-to-end auto-learning pipeline that completes entirely within a TypeScript environment (Node.js, Edge, etc.)**, without needing an external Python server or GPU.

## Overall Architecture

Auto-learning is achieved through the collaboration of the following components:

1. **`FeedbackCollector`**: Receives logs of user clicks and dwell times, and converts them into a learnable dataset (InfoNCE or Triplet format).
2. **`AdaptiveScheduler`**: Automatically triggers learning when a certain number of data points accumulate, while optimally decaying the Learning Rate.
3. **`InfoNCETrainer` / `TripletTrainer`**: A WASM-driven learning engine with a built-in Adam optimizer. It updates the weights that warp the vector space.
4. **Storage**: Saves the updated weights to a KVS like Redis or Cloudflare KV to reflect them in the next search.

## Implementation Steps

### 1. Initialization

Initialize the instances required for learning when the application starts or in the global scope of an edge environment.

```typescript
import { 
  FeedbackCollector, 
  AdaptiveScheduler, 
  InfoNCETrainer 
} from 'warpvector/ml';

// 1. Initialize the trainer (assuming 1536-dimensional vectors)
// We recommend InfoNCETrainer, which supports multiple Negatives and achieves high accuracy.
const trainer = new InfoNCETrainer(1536);

// 2. Configure the scheduler
// batchSize: How many logs to accumulate before triggering learning
// initialLearningRate: The initial learning rate
const scheduler = new AdaptiveScheduler(trainer, { 
  batchSize: 10,
  initialLearningRate: 0.01 
});

// 3. Configure the collector
// dwellThresholdMs: Consider it a "correct answer (positive)" if the user stays for this time (in milliseconds) or longer
const collector = new FeedbackCollector({ dwellThresholdMs: 3000 });
```

### 2. Collecting Logs and Recording Feedback

When a user performs a search and takes an action (like clicking) on the results, record the log in the collector.

```typescript
// Record the request information at the time of search and issue an impressionId.
const impressionId = collector.recordImpression({
  queryVector: rawQueryVector,            // The query vector at the time of search
  resultVectors: [docVec1, docVec2, docVec3], // List of vectors for the displayed search results
  timestamp: Date.now()
});

// Record when the user clicks the "1st result"
collector.recordFeedback({
  impressionId: impressionId,
  resultIndex: 0,   // The index of the clicked result
  type: "click"     // or "dwell", "skip"
});

// (Another search example) Record when the user views the "3rd result" for 5 seconds (dwell)
collector.recordFeedback({
  impressionId: anotherImpId,
  resultIndex: 2,
  type: "dwell",
  value: 5000       // Dwell time in milliseconds
});
```

### 3. Triggering Auto-Learning and Updating Weights

At an API endpoint that receives user actions, convert the data into learning data and feed it into the scheduler.

```typescript
async function handleUserAction() {
  // 1. Convert accumulated logs in the collector into InfoNCE learning data format
  const examples = collector.toInfoNCEExamples();
  if (examples.length === 0) return;

  // 2. Fetch the current "space-warping weights" from DB or KV (initial values for the first time)
  const currentWeights = await kvStore.get("intent_weights") || adapter.getIdentityWeights();

  // 3. Feed the data into the scheduler
  // * Internally, it checks if the number of logs has reached the batchSize (10 items).
  //   If it has, learning via the Adam optimizer runs on WASM.
  const updatedWeights = await scheduler.addFeedback(currentWeights, examples);

  // 4. Save to storage only if learning was triggered and weights were updated
  if (updatedWeights) {
    console.log("🔥 Auto-learning executed, weights updated!", scheduler.currentLearningRate);
    await kvStore.put("intent_weights", updatedWeights);
    
    // Clear the collector's buffer to prepare for the next batch
    collector.flush();
  }
}
```

### 4. Applying to Search (Inference)

The updated weights are reflected through the `IntentAdapter` when the next search request arrives. With just this, the vector space dynamically changes in a direction that users are more likely to click.

```typescript
import { IntentAdapter } from 'warpvector';

async function performSearch(queryVector) {
  // 1. Fetch the latest learned weights from storage
  const latestWeights = await kvStore.get("intent_weights");
  
  // 2. Apply to the adapter
  const adapter = new IntentAdapter(1536);
  adapter.addIntent("auto_learned", latestWeights);
  
  // 3. Warp (distort) the query vector
  const optimizedQuery = adapter.tune(queryVector, "auto_learned");
  
  // 4. Execute the search against the Vector DB
  return await pinecone.query({ vector: optimizedQuery, topK: 10 });
}
```

## Next Steps

- For a distributed learning architecture that centralizes the results learned across multiple edges (clients), please refer to [Federated Learning](./13-feedback-loop.md).
- For the mathematical background and types of base Trainers, please refer to [Trainers](./7-trainers.md).
