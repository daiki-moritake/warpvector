# Edge Computing Search Quickstart

This is a quickstart guide for building advanced search features (hybrid search, quantization, online learning) in edge environments (serverless environments) like Cloudflare Workers and Vercel Edge using Warpvector.

## 1. Installation
Because Warpvector is built zero-dependency and is TypeScript native, it runs in edge environments without any additional configuration.

```bash
npm install warpvector
```

## 2. Search Processing in Edge Workers (Vector Transformation + Hybrid Search)
This is an implementation example that receives a request within an edge function and completes everything from vector transformation to integrating hybrid search.

```typescript
import { WarpPipeline, rrf, QuantizationAdapter } from 'warpvector';

// Initialize the pipeline in the edge's global scope (to mitigate cold starts)
// WASM initialization also happens here.
const pipeline = new WarpPipeline(1536)
  .addIntent({ "tech_domain": techWeights })
  .setFinalStage("Quantization", new QuantizationAdapter({ type: "int8", dim: 1536 }));

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // 1. Prepare WASM (if already initialized, this is instantly skipped)
    await pipeline.init(); 

    const url = new URL(request.url);
    const query = url.searchParams.get("q");
    
    // 2. Call OpenAI or similar APIs to get the raw vector of the query
    const rawVector = await fetchEmbedding(query, env.OPENAI_API_KEY);

    // 3. Warp the vector to match the intent (tech_domain) and compress to Int8 on the edge
    const optimizedQuery = pipeline.run(rawVector, { intent: "tech_domain" });

    // 4. Send the search request to an external vector DB (like Pinecone)
    const pineconeQuery = pipeline.runAndFormat(
      optimizedQuery, 
      { format: "pinecone", topK: 10 }
    );
    const denseResultsPromise = fetchPinecone(pineconeQuery, env.PINECONE_KEY);
    
    // 5. Execute keyword search (Elasticsearch, etc.) simultaneously
    const sparseResultsPromise = fetchKeywordSearch(query);

    // 6. Wait for both search results
    const [denseResults, sparseResults] = await Promise.all([
      denseResultsPromise, sparseResultsPromise
    ]);

    // 7. Integrate hybrid search results on the edge using RRF (Reciprocal Rank Fusion)
    const finalResults = rrf([denseResults, sparseResults]);

    return new Response(JSON.stringify(finalResults), {
      headers: { "Content-Type": "application/json" }
    });
  }
};
```

## 3. Integrating Online Learning (Feedback Loops) at the Edge
This is a configuration to collect user click logs and perform real-time spatial optimization learning directly on the edge worker.

```typescript
import { FeedbackCollector, AdaptiveScheduler, TripletTrainer } from 'warpvector/ml';

// A collector that considers a dwell time of 3 seconds or more as positive
const collector = new FeedbackCollector({ dwellThresholdMs: 3000 });
const trainer = new TripletTrainer(1536);
// A scheduler that triggers automatic learning at a batch size of 5
const scheduler = new AdaptiveScheduler(trainer, { batchSize: 5 });

export async function handleUserAction(request: Request, env: Env) {
  const { queryVec, resultVecs, clickedIndex } = await request.json();
  
  // 1. Generate a Triplet (a correct/incorrect pair) from the click logs
  const impId = collector.recordImpression({
    queryVector: queryVec, 
    resultVectors: resultVecs, 
    timestamp: Date.now()
  });
  collector.recordFeedback({ 
    impressionId: impId, 
    resultIndex: clickedIndex, 
    type: "click" 
  });
  
  const examples = collector.toTripletExamples();
  
  // 2. Fetch the latest weights from edge storage
  const currentWeights = await env.KV_STORE.get("model_weights", "json"); 
  
  // 3. Once the batch size (5) is reached, execute high-speed learning with WASM and update weights
  const updatedWeights = await scheduler.addFeedback(currentWeights, examples);
  
  if (updatedWeights) {
    // 4. Save the updated weights to apply to the next search (warp) process
    await env.KV_STORE.put("model_weights", JSON.stringify(updatedWeights)); 
  }
  
  return new Response("OK");
}
```

## Next Steps
- For more advanced non-linear inference, see [Neural Networks](./2-neural-networks.md)
- For the `FedAvg` mechanism that aggregates locally learned weights for all users on the server side, see [Feedback Loop](./13-feedback-loop.md)
