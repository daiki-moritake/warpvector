# 量子化と圧縮 (Quantization)

高次元のベクトル（例: 1536次元の `Float32`）は、1ベクトルあたり 6,144バイト（約6KB）のメモリを消費します。100万件のベクトルをインメモリで保持・検索しようとすると、それだけで約6GBものメモリが必要になり、ブラウザやエッジ環境では到底扱えません。

WarpVector の `QuantizationAdapter` は、ベクトルの意味情報を極力保持したまま、データサイズを劇的に圧縮（量子化）する機能を提供します。

## 1. Int8 スカラー量子化 (メモリ1/4)

32ビット浮動小数点（Float32）の各次元の値を、-128 〜 127 の8ビット整数（Int8）にスケーリングして丸め込む手法です。
精度低下は非常に少なく（通常1〜2%未満のスコア誤差）、メモリ使用量を厳密に **4分の1** に削減します。

### 使い方

```typescript
import { QuantizationAdapter } from 'warpvector';

// Int8 量子化アダプタを作成
const int8Adapter = new QuantizationAdapter({ type: "int8", dim: 1536 });

// 1536次元の Float32Array
const floatVector = getFloat32Vector();

// 量子化の実行 (Int8Array が返る)
const int8Vec = int8Adapter.tune(floatVector);

// 検索時のスコア計算 (ドット積)
// Float32のコサイン類似度の近似として、Int8専用の高速ドット積を使用します
const similarityScore = QuantizationAdapter.int8DotProduct(int8Vec1, int8Vec2);
```

## 2. Binary (1-bit) 量子化 (メモリ1/32)

値を 0 または 1 の「ビット」にまで極限圧縮する手法です。値が 0 より大きければ 1、小さければ 0 と判定します。
さらに、32個のビットを 1つの数値（32ビット整数）に「パック」するため、メモリ使用量は驚異の **32分の1**（1536次元ならわずか 192バイト）になります。

### 使い方とハミング距離

Binary量子化されたベクトル同士の距離計算には、コサイン類似度ではなく「ハミング距離 (Hamming Distance)」を用います。これは、互いのビット列で「異なっているビットの数」を数えるものです。CPUレベルの XOR 演算とビットカウント（Popcount）で計算できるため、**数千万件の検索でも一瞬で完了**します。

```typescript
import { QuantizationAdapter } from 'warpvector';

// Binary 量子化アダプタを作成
const binaryAdapter = new QuantizationAdapter({ type: "binary", dim: 1536 });

// 量子化の実行 (Uint8Array としてパックされたデータが返る)
// 1536次元 -> 192個の Uint8 に圧縮される
const binVec = binaryAdapter.tune(floatVector);

// 検索時の距離計算 (ハミング距離)
// 距離が *小さい* ほど、類似度が高いことを意味します
const distance = QuantizationAdapter.hammingDistance(binVec1, binVec2);

// 応用: 距離(0〜1536)を類似度(0〜1)に変換したい場合
const similarity = 1.0 - (distance / 1536);
```

## 3. いつどちらを使うべきか？

- **Int8 量子化**: 検索の精度（Ranking）を極力落としたくない場合。メインの VectorDB のメモリ最適化として非常に有力です。
- **Binary 量子化**: とにかく超高速・省メモリで「候補の絞り込み（Retrieval）」を行いたい場合。100万件のデータから大まかに1000件の候補を Binary 検索で超高速に拾い上げ、その後、通常の Float32 や Int8 を使って正確なスコアで再ソート（Rerank）する構成（Two-stage Retrieval）がベストプラクティスです。
