# ハイブリッド検索フュージョン (Hybrid Search Fusion)

現代の検索システムのベストプラクティスは、WarpVector などの「ベクトル検索 (Dense Search: 意味的類似度)」と、Elasticsearch などの「キーワード検索 (Sparse Search/BM25: 単語の完全一致)」を組み合わせることです。
これを**ハイブリッド検索**と呼びます。

WarpVector は、全く異なる2つの検索エンジンのスコアを「公平に合体」させるための独立したフュージョン・ユーティリティを提供しています。

## 1. RRF (Reciprocal Rank Fusion)

RRF は、異なる検索システムから得られた結果の **スコアの絶対値（生の数値）を完全に無視し、「順位（Rank）」のみを用いて** 結果を統合するアルゴリズムです。
ベクトル検索のスコア（0.0 〜 1.0）とキーワード検索のスコア（10 〜 150など）のように、尺度が全く違うシステム同士を組み合わせる際に最も効果的です。

$$ RRF\_Score = \sum \frac{1}{k + Rank} $$
※ 一般的に定数 $k = 60$ がよく使われます。

### 使い方

```typescript
import { rrf } from 'warpvector';

// Vector DB (例: Pinecone) からの結果
const denseResults = [
  { id: "docA", score: 0.95 }, // 1位
  { id: "docB", score: 0.88 }, // 2位
  { id: "docC", score: 0.72 }  // 3位
];

// Keyword DB (例: Elasticsearch) からの結果
const sparseResults = [
  { id: "docB", score: 45.2 }, // 1位
  { id: "docD", score: 32.1 }, // 2位
  { id: "docA", score: 15.0 }  // 3位
];

// リストの配列を渡すだけで、順位ベースで統合された新しいリストを返します
const fusedResults = rrf([denseResults, sparseResults]);

console.log(fusedResults);
// docB と docA が上位にランクインします（両方のシステムで評価が高いため）
```

## 2. RSF (Relative Score Fusion)

RSF は、順位ではなく **「スコアの相対的な大きさ」** を重視したい場合に使用します。
各システムの結果リスト内でスコアを Min-Max 正規化 (0.0 〜 1.0 に変換) した上で、指定した重み（Weight）を掛けて足し合わせます。

「ベクトル検索の意味的な一致を重視しつつ (70%)、キーワードの完全一致も少し加味したい (30%)」といった微調整が可能です。

### 使い方

```typescript
import { rsf } from 'warpvector';

// 第1引数: 結果リストの配列
// 第2引数: 各リストに対する重みの配列
const fusedResults = rsf(
  [denseResults, sparseResults], 
  [0.7, 0.3] // Denseを70%、Sparseを30%の比重で統合
);
```

## メタデータの統合

`rrf` および `rsf` は、各アイテムが持つ `metadata` プロパティも自動的に引き継ぎます。同じIDのアイテムが複数のリストに存在し、それぞれ異なるメタデータを持っていた場合、それらは浅いマージ（Shallow Merge）されて最終結果に含まれます。
