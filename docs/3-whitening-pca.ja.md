# オンライン等方化・PCA (Online Whitening & PCA)

事前学習済みの大規模言語モデル（例：OpenAI `ada-002` や Cohere）が生成するベクトルには、特有の「空間の偏り（異方性）」が存在します。すべてのベクトルが特定の方向（コーン状の狭い領域）を向いてしまい、一見無関係な文章同士でもコサイン類似度が `0.80` 以上になってしまう現象です。

WarpVectorの `WhiteningAdapter` は、**Oja's Rule**（オンライン主成分分析アルゴリズム）を用いて、ベクトルのストリームデータからこの「偏りの方向（第一主成分）」を自律的に学習し、検索時に動的にその偏りを取り除きます（等方化 / Whitening）。

## 1. WhiteningAdapter の仕組み

1. **学習 (Update)**: ベクトルをシステムに送るたびに、現在の「一番分散が大きい方向（偏り）」をストリーミング学習で推測します。全データをメモリに保持して特異値分解（SVD）を行う必要がないため、メモリ使用量は極小です。
2. **適用 (Tune)**: 検索時や保存時に、学習した「偏りの方向」の成分をベクトルから引き算（直交射影）します。

これにより、すべてのベクトルが空間全体に球状（等方的）に分散するようになり、コサイン類似度の解像度が劇的に向上します（例：関連するものは `0.9`、無関係なものは `0.1` と明確に分かれるようになります）。

## 2. 基本的な使い方

```typescript
import { WhiteningAdapter } from 'warpvector';

// 1536次元のベクトルに対して、上位1つの主成分（偏り）を学習して除去する設定
// learningRate はオンライン学習の更新率
const adapter = new WhiteningAdapter(1536, { 
  learningRate: 0.01, 
  numComponents: 1 
});

// アプリケーションでベクトルを生成・保存するついでに update を呼ぶ
function onDocumentAdded(vector: number[]) {
  // 偏りの方向を少しずつ学習
  adapter.update(vector); 
}

// データ検索時（またはDB保存時）に tune を呼ぶ
function onSearch(queryVector: number[]) {
  // 偏り成分を除去（無意味に高い類似度を相殺）
  const whitenedQuery = adapter.tune(queryVector);
  
  // whitenedQuery を使って Vector DB を検索...
}
```

## 3. なぜバッチPCAではなくオンラインPCAなのか？

通常のPCA（主成分分析）は、数百万件のベクトルデータを一度に全てメモリにロードし、巨大な共分散行列を計算する必要があります。これはNode.jsやエッジ環境（Cloudflare Workers等）では現実的ではありません。

WarpVector のオンラインPCAは、ベクトルが1つ来るたびに主成分ベクトルをわずかに更新するため：
- メモリ使用量はベクトル1本分（数KB）に固定。
- データの傾向が時間経過で変化（コンセプト・ドリフト）しても、自動的に追従して補正。
- VectorDBへデータを追加するパイプラインに、文字通り `adapter.update()` を1行挟むだけで済みます。

## 4. 複数成分の除去

`numComponents` を増やすことで、第1主成分だけでなく、第2、第3のノイズ成分も除去できます。ただし、多く除去しすぎると文章の本来の意味（シグナル）まで失われるため、通常は `1` または `2` が推奨されます。

```typescript
const adapter = new WhiteningAdapter(1536, { numComponents: 2 });
```
