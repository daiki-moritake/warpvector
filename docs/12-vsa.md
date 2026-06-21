
# 超次元計算 / VSA (Vector Symbolic Architecture)

## 概要

`VsaAdapter` は、ベクトル・シンボリック・アーキテクチャ (VSA) / 超次元計算 (Hyperdimensional Computing) の演算を提供します。

ベクトル同士を論理的・数学的に結合（バインド）したり束ねたり（バンドル）することで、1つの密ベクトルの中にキーと値（メタデータなど）を埋め込み、検索空間上でそのまま演算を行えるようにします。

## 3つの基本演算

### 1. バンドル (Bundle / Superposition)

複数のベクトルを足し合わせ（重ね合わせ）、1つのベクトルに統合します。「A と B の両方の概念を含む」ベクトルを作る際に使います。

```typescript
import { VsaAdapter } from '@warpvector/extras';

// 「科学」と「技術」の概念を統合
const sciTech = VsaAdapter.bundle([scienceVec, technologyVec]);
// sciTech は両方にコサイン類似度が高い
```

### 2. バインド (Bind / Hadamard Product)

アダマール積（要素ごとの積）により、2つのベクトルを「結合」します。キー（ユーザーID）と値（好み）を掛け合わせて、特有のベクトルを生成します。

```typescript
// ユーザーIDベクトルと好みベクトルを結合
const bound = VsaAdapter.bind(userIdVec, preferenceVec);
```

### 3. アンバインド (Unbind)

バインドされたベクトルから、片方のキーを使って元の値を取り出します。

```typescript
// ユーザーIDベクトルをキーとして、好みベクトルを抽出
const recovered = VsaAdapter.unbind(bound, userIdVec);
// recovered ≈ preferenceVec (近似的に復元)
```

## Binary VSA (XOR演算)

`QuantizationAdapter` で 1-bit (Binary) 量子化された `Uint8Array` ベクトルに対する超高速な VSA 演算です。XOR 演算により、極小メモリでの超高速処理が可能です。

### bindBinary / unbindBinary

XOR の自己逆性 (`A ^ B ^ B = A`) を利用して、バインドとアンバインドを実行します。

```typescript
const binaryBound = VsaAdapter.bindBinary(binKey, binValue);
const binaryRecovered = VsaAdapter.unbindBinary(binaryBound, binKey);
// binaryRecovered === binValue
```

### bundleBinary (多数決投票)

複数のバイナリベクトルを重ね合わせます。各ビット位置で 1 と 0 の多数決 (Majority Vote) により最終ビットを決定します。

```typescript
const merged = VsaAdapter.bundleBinary([bin1, bin2, bin3]);
```

## ユースケース

### メタデータの埋め込み検索

ベクトルにメタデータを埋め込み、検索と属性フィルタリングを同時に実行する例:

```typescript
// 1. 各属性をキーベクトルとバインド
const categoryBound = VsaAdapter.bind(categoryKeyVec, categoryValueVec);
const priceBound = VsaAdapter.bind(priceKeyVec, priceRangeVec);

// 2. 元のベクトルとメタデータをバンドル
const enrichedDoc = VsaAdapter.bundle([
  documentVec,
  categoryBound,
  priceBound,
]);

// 3. 検索時に特定のメタデータを抽出
const extractedCategory = VsaAdapter.unbind(enrichedDoc, categoryKeyVec);
```

## API

### 密ベクトル演算

| メソッド | 説明 |
|---|---|
| `VsaAdapter.bundle(vectors, options?)` | 複数ベクトルの重ね合わせ（L2正規化付き） |
| `VsaAdapter.bind(vec1, vec2, options?)` | アダマール積によるバインド |
| `VsaAdapter.unbind(boundVec, keyVec, options?)` | 要素ごとの除算によるアンバインド |

### バイナリベクトル演算

| メソッド | 説明 |
|---|---|
| `VsaAdapter.bindBinary(bin1, bin2)` | XOR によるバイナリバインド |
| `VsaAdapter.unbindBinary(boundBin, keyBin)` | XOR の自己逆性による抽出 |
| `VsaAdapter.bundleBinary(bins)` | 多数決投票によるバイナリバンドル |

### VsaOptions

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `shouldNormalize` | `boolean` | `true` | 結果をL2正規化するか |
