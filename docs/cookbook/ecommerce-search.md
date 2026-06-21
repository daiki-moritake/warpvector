# Cookbook: E-commerce Intent Search

In e-commerce, the same search query can mean completely different things based on the user's current browsing context or past behavior. For example, if a user searches for "shoes", do they want formal leather shoes, or running sneakers?

Instead of maintaining separate indexes or passing complex metadata filters, you can use `IntentAdapter` to dynamically warp the vector space.

## Implementation

```typescript
import { IntentAdapter } from 'warpvector';

// 1. Initialize the adapter with your embedding dimension (e.g. OpenAI 1536)
const adapter = new IntentAdapter(1536);

// 2. Define intents by providing a transformation matrix and bias.
// (These are usually pre-calculated or learned via feedback loop)
adapter.addIntent("formal", { matrix: formalMatrix, bias: formalBias });
adapter.addIntent("sports", { matrix: sportsMatrix, bias: sportsBias });

// 3. User makes a search query
const rawQueryVector = await getOpenAIEmbedding("shoes");

// 4. Check user context (e.g., from session, previous clicks, or current category)
const userContext = getUserContext(req);

let warpedQueryVector = rawQueryVector;

if (userContext.preference === "formal_wear") {
  // Warp the vector towards formal concepts
  warpedQueryVector = adapter.tune(rawQueryVector, "formal");
} else if (userContext.preference === "athletic") {
  // Warp the vector towards sports concepts
  warpedQueryVector = adapter.tune(rawQueryVector, "sports");
}

// 5. Search the database with the warped vector
const results = await vectorDb.search(warpedQueryVector);
```

## How it works
The `adapter.tune()` function performs a highly optimized matrix multiplication using WASM. It takes less than 1µs. The resulting vector will be mathematically closer to items in your database that match the intended category, improving search relevance without retraining your core embedding model.
