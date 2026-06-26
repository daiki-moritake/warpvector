# Hacker News / Reddit Launch Post Draft

## Title Ideas
- Show HN: WarpVector – Zero-dependency runtime vector space transformation in TS
- Show HN: Reduce your Pinecone bill by 96% with TS vector quantization
- WarpVector: A fast, zero-dependency middleware for vector embeddings in TypeScript

## Body Text

Hey HN,

I'm Daiki, and I’m building **WarpVector** (https://github.com/daiki-moritake/warpvector).

When building RAG applications, we often run into two big problems:
1. **Vector DBs are expensive.** Storing 1 million 1536-dimensional float32 vectors takes about 6GB of RAM. If you use a hosted vector DB, this quickly becomes a significant monthly cost.
2. **Static embeddings lack context.** The word "Apple" might mean the fruit or the company. Traditional vector search can't adapt to the user's intent without complex metadata filtering or expensive LLM fine-tuning.

WarpVector is a lightweight, zero-dependency TypeScript middleware that sits between your embedding model and your vector database.

It solves these problems by applying fast, in-memory affine transformations (powered by WASM) to your vectors at query time:

- **Quantization (Int8 & Binary):** Shrink your float32 vectors into Int8 or Binary vectors before sending them to your DB. Binary quantization shrinks vectors by up to 96.9% (6GB -> 192MB) while preserving 0.9999+ cosine similarity rankings.
- **Intent Warping:** Dynamically "warp" the vector space towards a specific domain (e.g. "technical" vs "business") based on user context. No LLM retraining needed.
- **Edge Ready:** Because it's written in TS and WASM, it runs seamlessly on Cloudflare Workers and Vercel Edge with ~1µs latency per vector.

**Interactive Demo:** We built a playground that runs the actual WASM library in your browser to visualize how the vector space warps in real-time. Check it out here: https://daiki-moritake.github.io/warpvector/

We also have a CLI if you want to scaffold a Next.js or Edge application quickly:
`npx create-warpvector-app`

I'd love to hear your feedback on the API design, the WASM implementation, or any use cases you can think of! Happy to answer any questions.

Repo: https://github.com/daiki-moritake/warpvector
Docs: https://github.com/daiki-moritake/warpvector/tree/main/docs
