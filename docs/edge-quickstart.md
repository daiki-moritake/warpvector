# Edge Computing Search Quickstart

Warpvector を用いて、Cloudflare Workers や Vercel Edge などのエッジ環境（サーバーレス環境）で、高度な検索機能（ハイブリッド検索、量子化、オンライン学習）を構築するためのクイックスタートガイドです。

## 1. インストール
Warpvectorはゼロ依存でTypeScriptネイティブに作られているため、エッジ環境でも追加の設定なしに動作します。

```bash
npm install warpvector
```

## 2. エッジワーカーでの検索処理（ベクトル変換＋ハイブリッド検索）
エッジ関数内でリクエストを受け取り、ベクトル変換からハイブリッド検索の統合までを完結させる実装例です。

```typescript
import { WarpPipeline, rrf, QuantizationAdapter } from 'warpvector';

// パイプラインはエッジのグローバルスコープで初期化（コールドスタート対策）
// WASMの初期化もここで行われます。
const pipeline = new WarpPipeline(1536)
  .addIntent({ "tech_domain": techWeights })
  .setFinalStage("Quantization", new QuantizationAdapter({ type: "int8", dim: 1536 }));

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // 1. WASMの準備（すでに初期化済みの場合は即座にスキップされます）
    await pipeline.init(); 

    const url = new URL(request.url);
    const query = url.searchParams.get("q");
    
    // 2. OpenAI 等のAPIを叩いてクエリの生のベクトルを取得
    const rawVector = await fetchEmbedding(query, env.OPENAI_API_KEY);

    // 3. エッジ上でベクトルを意図（tech_domain）に合わせてワープ＆Int8に圧縮
    const optimizedQuery = pipeline.run(rawVector, { intent: "tech_domain" });

    // 4. 外部のベクトルDB（Pinecone等）に検索リクエストを送信
    const pineconeQuery = pipeline.runAndFormat(
      optimizedQuery, 
      { format: "pinecone", topK: 10 }
    );
    const denseResultsPromise = fetchPinecone(pineconeQuery, env.PINECONE_KEY);
    
    // 5. 同時にキーワード検索（Elasticsearch等）を実行
    const sparseResultsPromise = fetchKeywordSearch(query);

    // 6. 両方の検索結果を待ち合わせ
    const [denseResults, sparseResults] = await Promise.all([
      denseResultsPromise, sparseResultsPromise
    ]);

    // 7. エッジ上で RRF (Reciprocal Rank Fusion) によるハイブリッド検索結果の統合
    const finalResults = rrf([denseResults, sparseResults]);

    return new Response(JSON.stringify(finalResults), {
      headers: { "Content-Type": "application/json" }
    });
  }
};
```

## 3. エッジでのオンライン学習（フィードバックループ）の組み込み
ユーザーのクリックログを収集し、エッジワーカー上で直接リアルタイムに空間最適化の学習を行う設定です。

```typescript
import { FeedbackCollector, AdaptiveScheduler, TripletTrainer } from 'warpvector/ml';

// 滞在時間3秒以上をポジティブとみなすコレクター
const collector = new FeedbackCollector({ dwellThresholdMs: 3000 });
const trainer = new TripletTrainer(1536);
// バッチサイズ5で自動学習を発火させるスケジューラー
const scheduler = new AdaptiveScheduler(trainer, { batchSize: 5 });

export async function handleUserAction(request: Request, env: Env) {
  const { queryVec, resultVecs, clickedIndex } = await request.json();
  
  // 1. クリックログからTriplet（正解・不正解の組）を生成
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
  
  // 2. 最新の重みをエッジのストレージから取得
  const currentWeights = await env.KV_STORE.get("model_weights", "json"); 
  
  // 3. バッチサイズ(5)に達した場合、WASMで高速に学習を実行し、重みを更新
  const updatedWeights = await scheduler.addFeedback(currentWeights, examples);
  
  if (updatedWeights) {
    // 4. 更新された重みを保存し、次回の検索（ワープ）処理に反映
    await env.KV_STORE.put("model_weights", JSON.stringify(updatedWeights)); 
  }
  
  return new Response("OK");
}
```

## 次のステップ
- より高度な非線形推論については [Neural Networks](./2-neural-networks.md) を参照
- ローカルで学習した重みをサーバー側で全ユーザー分集約する `FedAvg` の仕組みについては [Feedback Loop](./13-feedback-loop.md) を参照
