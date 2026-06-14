# Advanced Usage (高度な使い方)

`warpvector` に備わっているより高度なユースケースや、パフォーマンスを極限まで引き出すための機能を紹介します。

---

## 1. 非線形活性化関数 (Non-linear Activations)

単純なアフィン変換（線形）だけでは表現しきれない複雑な概念の切り分けを行うため、`tune` などの関数にはオプションで **非線形活性化関数** を指定できます。

```typescript
// 負の値を0に切り捨てる (ReLU)
const reluVector = adapter.tune(baseVector, "riskAnalysis", "relu");

// 空間を 0.0 〜 1.0 の範囲に圧縮する (Sigmoid)
const sigmoidVector = adapter.tune(baseVector, "riskAnalysis", "sigmoid");

// 空間を -1.0 〜 1.0 の範囲に圧縮する (Tanh)
const tanhVector = adapter.tune(baseVector, "riskAnalysis", "tanh");
```

---

## 2. 自己アテンション型動的ブレンド (Auto-blending)

ユーザーが「リスク分析を70%、経済影響を30%で検索したい」と明示しなくても、入力されたクエリベクトル自身の意味合いから、自動で最も適したインテント比率を推論・合成することができます。

これを利用するには、各インテントの初期化時に `routingVector`（そのインテントを代表するベクトル）を設定します。

```typescript
const adapter = new IntentAdapter({
  intentA: {
    matrix: [...], bias: [...],
    routingVector: [1.0, 0.0, 0.0] // intentA を表す代表ベクトル
  },
  intentB: {
    matrix: [...], bias: [...],
    routingVector: [0.0, 1.0, 0.0] // intentB を表す代表ベクトル
  }
});

// クエリベクトルの中身に基づいて、自動的に intentA と intentB の比率を
// コサイン類似度 + Softmax で計算し、ブレンド変換を適用する
const autoTuned = adapter.tuneAutoBlended(queryVector);
```

---

## 3. WASM / SIMD による超高速バッチ処理

Pinecone などのベクトルデータベースから10,000件のデータを取得し、フロントエンドやエッジサーバーで再ランキング（リランキング）するようなユースケースにおいて、`for`ループによる単一変換はボトルネックになる可能性があります。

`warpvector` では `tuneBatch` や `tuneBatchBlended` を呼び出すと、内部で自動的に WebAssembly (WASM) モジュールが呼び出されます。

```typescript
// ベクトルの配列（バッチ）
const batchVectors = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  // ... 10,000件のデータ
];

// 内部でWASMの共有メモリに一括転送され、Float32による最適化された計算が実行される
const tunedBatch = adapter.tuneBatch(batchVectors, "riskAnalysis");
```
ユーザー側でWASMのロードやメモリ管理を意識する必要はありません。ブラウザでも、Node.jsでも、Cloudflare Workersでもシームレスに動作します。

---

## 4. LoRA アダプターによる超高次元ベクトルの最適化

OpenAIの `text-embedding-3-small` (1536次元) のような超高次元ベクトルの場合、通常の `IntentAdapter` では `1536 x 1536`（約2.36百万パラメータ）のフルマトリックスが必要となり、メモリと計算量が増加します。

`LoraIntentAdapter` は、このフルマトリックスを「低ランク」の行列A・行列Bに分解（例：ランク16）することで、表現力を維持したままパラメータ数を 1536 * 16 * 2 = 49,152（約98%削減）に圧縮します。

```typescript
import { LoraIntentAdapter } from 'warpvector';

// 次元数 1536, ランク 16 で初期化
const loraAdapter = new LoraIntentAdapter(1536, 16);

loraAdapter.addIntent("myContext", {
  matrixA: [...], // 1536 x 16
  matrixB: [...], // 16 x 1536
  bias: [...]     // 1536
});

const tuned = loraAdapter.tune(baseVector, "myContext");
```
大規模言語モデル（LLM）のファインチューニングで使われる手法を、そのままインメモリのベクトル検索空間に応用しています。
