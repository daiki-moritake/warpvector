# 14. 意味のソフトな白色化 (Soft Whitening)

`SoftWhiteningAdapter` は、大規模言語モデル（LLM）の埋め込みベクトルに特有の「一般的な文脈への偏り（Semantic Diffusion）」を、**主成分の指数減衰（Principal Component Attenuation）**を用いて解消する強力なコンポーネントです。

## なぜ「Soft Whitening」が必要なのか？

OpenAIの `text-embedding-ada-002` や `text-embedding-3-small` のような汎用モデルは、文章の文脈を広く捉える能力に長けています。しかし、特定のドメイン（医療、法律、社内用語など）において検索を行うと、以下のような問題が発生します。

*   **コンテキストの過剰な混合:** ニッチなキーワードで検索しても、それに類似した一般的な言葉に引っ張られ、ベクトルが「ぼやけて」しまいます。
*   **「なんでも似ている」問題:** 分散の大きい主成分（頻出単語や一般的な文体成分）が類似度計算を支配してしまい、本来見つけたかった鋭い意図（Sharp Source）が埋もれてしまいます。

これを解決するために、分散が大きい（一般的な文脈で広がっている）主成分に対してソフトな減衰（Attenuation）をかけることで、特有のニッチな意味を際立たせるのが、** Soft Whitening ** です。

---

## 仕組み

`SoftWhiteningAdapter` は、ストリーミングデータからオンラインで「上位 $K$ 個の主成分」とその「分散（固有値 $\lambda_k$）」をトラッキングします。

そして推論時に、シャープネス・パラメータ $\tau$ (tau) を用いて、分散が大きい成分ほど強く減衰（フィルタリング）させます。
数式としては、主成分方向の射影成分に対して以下の指数減衰を適用します。

$$ x'_{proj} = e^{-\tau \lambda_k} x_{proj} $$

（内部実装としては、以下の減衰係数を引いています）
$$ \text{Attenuation}(k) = 1 - \exp(-\tau \lambda_k) $$

- 固有値 $\lambda_k$ が大きい（よくある一般的な文脈に拡散している）方向は、大きく削られます。
- 固有値が小さい（特異な、シャープな意味を持つ）方向はほぼ保持されます。

---

## 実装例

```typescript
import { SoftWhiteningAdapter } from 'warpvector';

// ベクトル次元数 1536 (OpenAI ada-002 など)
// 固有空間の分散をトラッキングし、ソフトな白色化フィルタを適用するアダプタ
const adapter = new SoftWhiteningAdapter(1536, {
  learningRate: 0.01,
  numComponents: 5,   // 上位5つの成分をトラッキング
  tau: 2.0,           // シャープネスの強さ。大きいほど強く抑制する。
  normalizeOutput: true // 出力をL2正規化する (Cosine Similarity前提の場合はtrue)
});

// --- 1. ストリーミング学習 (Online Tracking) ---
// ユーザーのクエリやコーパスのドキュメント等から、空間の広がり方を学習します。
adapter.update(vectorA);
adapter.update(vectorB);
adapter.update(vectorC);

// --- 2. シャープニング (Inference) ---
// 検索時にコンテキストの濁りを解消し、本来の鋭い意味空間へ変換します。
const sharpVector = adapter.tune(queryVector);

// --- 3. バッチ処理 (最適化) ---
// 複数のベクトル（例えばDBのインデックス作成時）は `tuneBatch` で高速一括処理が可能です。
const sharpVectors = adapter.tuneBatch([vector1, vector2, vector3]);
```

---

## パラメータのチューニングガイド

`SoftWhiteningConfig` で指定できるパラメータの解説とベストプラクティスです。

### 1. `tau` (シャープネスの強さ)
最も重要なパラメータです。デフォルトは `1.0` です。
- `tau = 0`: 一切の補正を行いません（無効化）。
- `tau = 0.5 ~ 2.0`: マイルドなシャープニング。一般的な RAG (Retrieval-Augmented Generation) でコンテキストのノイズを減らしたい場合に推奨されます。
- `tau = 5.0 ~`: 非常に強いシャープニング。上位主成分がほぼ完全に除去されます。
- `tau → ∞`: `WhiteningAdapter` と同等の挙動（完全な直交化）に漸近します。

### 2. `numComponents` (トラッキングする主成分数)
通常は `1` 〜 `10` の間で設定します。デフォルトは `5` です。
大きくしすぎると、意味のある重要なニッチ成分まで削ってしまう可能性があるため、コーパスの多様性に合わせて調整します。

### 3. `normalizeOutput` (出力の正規化)
デフォルトは `true` です。
成分方向の長さが削られるため、ベクトルの全体ノルム（長さ）が 1.0 から外れます。
Pinecone, Qdrant, pgvector などで **Cosine Similarity** (内積ではなくコサイン) を前提としている場合は、長さを 1.0 に戻す必要があるため、必ず `true` に設定してください。

---

## `WhiteningAdapter` との違い

`WarpVector` には空間的偏りを除去する `WhiteningAdapter` も存在します。
これらは似た目的を持っていますが、以下のように使い分けます。

| 特徴 | WhiteningAdapter | SoftWhiteningAdapter |
| :--- | :--- | :--- |
| **アプローチ** | 異方性（偏り）を完全に「除去」する | 分散の大きさに応じて「滑らかに減衰」させる |
| **情報の損失** | 上位成分の情報は完全に失われる | 上位成分もわずかに残るため、破綻しにくい |
| **パラメータ** | `numComponents` のみ | `numComponents` と `tau` (滑らかさの調整が可能) |
| **適した用途** | 言語モデル固有の強烈な固定バイアスを消したい時 | 文脈が混ざってしまっている検索クエリをシャープにしたい時 |

より高度でデリケートなチューニング（検索意図を壊しすぎない調整）を求めるプロダクトにおいては、`SoftWhiteningAdapter` の導入を強く推奨します。
