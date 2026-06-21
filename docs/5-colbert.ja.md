# Late Interaction / ColBERT

一般的なベクトル検索では、文書全体やクエリ全体の意味を「1つのベクトル（例: 1536次元）」に圧縮して比較します（Dense Retrieval）。しかしこれでは、長い文書に含まれる特定の「細かいニュアンス」や「単語の組み合わせ」が圧縮の過程で潰れてしまうという課題があります。

WarpVector の `ColbertAdapter` は、**Late Interaction (ColBERT アーキテクチャ)** をエッジ環境で実現するためのアダプタです。
クエリと文書を1つのベクトルではなく、「各トークン（単語）ごとのベクトルの集合（マトリクス）」として保持し、検索の最後に緻密な総当り照合（MaxSim）を行います。

## 1. Late Interaction (MaxSim) の仕組み

1. クエリ "A B C" を `[Vec_A, Vec_B, Vec_C]` のように複数のベクトルとして表現します。
2. ドキュメント "X Y Z ..." も `[Vec_X, Vec_Y, Vec_Z, ...]` として表現します。
3. クエリの各トークンについて、ドキュメント内の全トークンとの類似度を計算し、**「最も高い類似度 (MaxSim)」** を見つけます。
4. 全てのクエリトークンの MaxSim を合計したものが、最終的なドキュメントのスコアになります。

これにより、「Aという概念」がドキュメント内のどこかに存在し、「Bという概念」が別のどこかに存在していれば、それが的確にマッチングされます。単一ベクトルでは絶対に不可能な高精度の検索体験を提供します。

## 2. WASM による超高速化

この「総当り照合」は、TypeScript/JavaScript で実行すると絶望的に遅い（何重もの for ループと数万回のドット積が必要になるため）という致命的な欠点があります。
WarpVector は、この MaxSim 演算を **WebAssembly (WASM)** のフラットメモリ上で実行し、ループのアンローリングとSIMDライクな処理によって爆速化しています。

## 3. 基本的な使い方

```typescript
import { ColbertAdapter } from 'warpvector';

const adapter = new ColbertAdapter();

// クエリとドキュメントは、それぞれ「トークンベクトルの平坦化配列」として用意します
// 例: 32次元のトークンが5個ある場合、32 * 5 = 160要素の Float32Array
const queryTokens = getQueryTokenMatrix(); 

const doc1Tokens = getDocTokenMatrix(doc1);
const doc2Tokens = getDocTokenMatrix(doc2);
const doc3Tokens = getDocTokenMatrix(doc3);

const documents = [doc1Tokens, doc2Tokens, doc3Tokens];
const dimension = 32; // トークン1つあたりの次元数

// WASM上で超高速に全ドキュメントに対する MaxSim を計算し、スコア順にランク付け
const rankedResults = adapter.rank(queryTokens, documents, dimension);

console.log(rankedResults);
// 出力例:
// [
//   { index: 1, score: 12.45 }, // doc2 が1位
//   { index: 0, score:  9.12 }, // doc1 が2位
//   { index: 2, score:  4.33 }  // doc3 が3位
// ]
```

## 4. ユースケース

ColBERTアーキテクチャによるLate Interactionは、特に **RAG (Retrieval-Augmented Generation)** において圧倒的な威力を発揮します。

ユーザーからの長文で複雑な質問や、「〇〇について、××の観点から教えて」といった複数の条件が絡むクエリに対して、単一ベクトルの検索ではどうしても「全体的にぼんやり似ている別の文書」が引っかかりがちですが、ColBERTを用いれば「〇〇」と「××」の両方のトークンが文中に出現する文書をピンポイントで引き当てることができます。
