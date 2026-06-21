# Cookbook: エッジ（Cloudflare Workers）でのWarpVectorの実行

WarpVectorはZero-dependency（依存関係なし）で設計されており、内部的にWASMを利用しているため、Cloudflare WorkersやVercel Edge Functionsなどのエッジコンピューティング環境での実行に最適です。

エッジでベクトル変換を実行することで、地理的にユーザーに最も近い場所で検索クエリをパーソナライズでき、すべてのリクエストを中央のPythonベースのMLサーバーにルーティングする場合と比較してレイテンシを大幅に削減できます。

## 実装

```typescript
// src/index.ts
import { IntentAdapter } from 'warpvector';

export interface Env {
  // ここにCloudflareのバインディングを追加します
  VECTOR_DB_API_KEY: string;
}

// リクエスト間で再利用するために、fetchハンドラの外でアダプタを初期化できます
// 注意: 実際のアプリでは、KVやR2ストアからこれらの行列をロードします
const adapter = new IntentAdapter(1536);
// adapter.addIntent("premium_user", { matrix: ..., bias: ... });

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    try {
      // 1. ユーザーからクエリを取得
      const { query, isPremium } = await request.json();
      
      // 2. OpenAI APIから生の埋め込みベクトルを取得
      const openAiRes = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${env.OPENAI_API_KEY}` },
        body: JSON.stringify({ input: query, model: "text-embedding-3-small" })
      });
      const rawVector = (await openAiRes.json()).data[0].embedding;
      
      // 3. ユーザーのティアに基づいて、エッジでベクトルをワープ（変換）
      let finalVector = rawVector;
      if (isPremium) {
        // サブミリ秒のWASM実行
        finalVector = adapter.tune(rawVector, "premium_user");
      }
      
      // 4. ワープされたベクトルをベクトルDBに送信
      // ... 検索ロジック ...
      
      return new Response(JSON.stringify({ results: [] }), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (e) {
      return new Response(e.message, { status: 500 });
    }
  },
};
```

## パフォーマンスについての注記
WarpVectorの `IntentAdapter` は **1ベクトルあたり約1.1〜3.8マイクロ秒**で実行されます。このオーバーヘッドはエッジのリクエストライフサイクルにおいては実質的に知覚できないレベルであり、APIエンドポイントに遅延を追加することなく、強力な動的パーソナライゼーションを提供します。
