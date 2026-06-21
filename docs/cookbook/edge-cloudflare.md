# Cookbook: Running WarpVector at the Edge (Cloudflare Workers)

WarpVector is designed to be zero-dependency and utilizes WASM under the hood, making it perfectly suited for Edge computing environments like Cloudflare Workers or Vercel Edge Functions.

Running vector transformations at the edge allows you to personalize search queries closest to the user geographically, significantly reducing latency compared to routing every request through a central python-based ML server.

## Implementation

```typescript
// src/index.ts
import { IntentAdapter } from 'warpvector';

export interface Env {
  // Add your Cloudflare bindings here
  VECTOR_DB_API_KEY: string;
}

// You can initialize the adapter outside the fetch handler to reuse it across requests
// Note: In a real app, you would load these matrices from your KV or R2 store
const adapter = new IntentAdapter(1536);
// adapter.addIntent("premium_user", { matrix: ..., bias: ... });

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    try {
      // 1. Get the query from the user
      const { query, isPremium } = await request.json();
      
      // 2. Fetch the raw embedding from OpenAI API
      const openAiRes = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${env.OPENAI_API_KEY}` },
        body: JSON.stringify({ input: query, model: "text-embedding-3-small" })
      });
      const rawVector = (await openAiRes.json()).data[0].embedding;
      
      // 3. Warp the vector at the edge based on user tier
      let finalVector = rawVector;
      if (isPremium) {
        // Sub-millisecond WASM execution
        finalVector = adapter.tune(rawVector, "premium_user");
      }
      
      // 4. Send the warped vector to your Vector DB
      // ... search logic ...
      
      return new Response(JSON.stringify({ results: [] }), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (e) {
      return new Response(e.message, { status: 500 });
    }
  },
};
```

## Performance Note
WarpVector's `IntentAdapter` executes in **~1.1-3.8 µs per vector**. This overhead is virtually unnoticeable in an edge request lifecycle, adding no perceptible latency to your API endpoints while providing powerful dynamic personalization.
