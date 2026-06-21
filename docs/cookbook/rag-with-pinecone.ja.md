# Cookbook: Pineconeを用いたコスト効率の高いRAG

PineconeやQdrantなどのマネージドベクトルデータベースに何百万ものベクトルを保存すると、これらのデータベースはベクトルをメモリ上に保持するため、あっという間にコストが高騰する可能性があります。

WarpVectorを使用すると、データベースに送信する**前**に `Float32` ベクトルを `Int8` または `Binary` 形式に圧縮でき、ストレージとメモリのコストを最大96%削減できます。

## 実装

```typescript
import { Pinecone } from '@pinecone-database/pinecone';
import { QuantizationAdapter } from 'warpvector/extras';

// 1. Binary量子化アダプタを初期化
const quantizer = new QuantizationAdapter({ type: 'binary', dim: 1536 });
const pc = new Pinecone();
const index = pc.index('my-rag-index');

/**
 * ドキュメントのインデックス作成
 */
async function indexDocument(id: string, text: string) {
  // OpenAIから標準的なFloat32の埋め込みを取得
  const rawVector = await getOpenAIEmbedding(text);
  
  // Binaryに圧縮 (1536 float32 -> 1536 bits = 192 bytes)
  const compressedVector = quantizer.tune(rawVector);
  
  // Pineconeに保存
  await index.upsert([{
    id,
    values: Array.from(compressedVector), // Uint8Arrayを通常の配列に変換
    metadata: { text }
  }]);
}

/**
 * 検索 (RAG)
 */
async function search(query: string) {
  const rawQueryVector = await getOpenAIEmbedding(query);
  
  // 同じ量子化器を使用してクエリベクトルを圧縮
  const compressedQueryVector = quantizer.tune(rawQueryVector);
  
  // 圧縮されたベクトルを使用してPineconeを検索
  // Pineconeのインデックスがハミング距離(hamming)または内積(dotproduct)に設定されていることを確認してください
  const results = await index.query({
    vector: Array.from(compressedQueryVector),
    topK: 5,
    includeMetadata: true
  });
  
  return results;
}
```

## ROI (投資対効果)
- **Float32**: 1,000,000ベクトル × 1536次元 × 4バイト = 約 6.14 GB のRAM
- **Binary**: 1,000,000ベクトル × 1536次元 × 0.125バイト = 約 192 MB のRAM

データベースストレージを約96.8%節約でき、これが直接的にインフラストラクチャの月額請求額の大幅な削減につながります。
