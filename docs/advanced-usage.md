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

---

## 5. IntentTrainer による意図行列の自動学習 (Auto-training)

巨大な変換行列を手動で定義することは困難です。`IntentTrainer` を使えば、「この入力ベクトルには、この結果ベクトルを返してほしい」という少量のサンプルデータから、**最適な意図行列（IntentWeights）を自動的に学習**することができます。

```typescript
import { IntentTrainer } from 'warpvector';

// 1. 次元数を指定してトレーナーを初期化
const trainer = new IntentTrainer(1536);

// 2. 学習データ（正例）を追加
trainer.addExample({
  input: [...],  // 検索クエリのベクトル
  target: [...]  // 理想的なドキュメントのベクトル
});

// 3. 確率的勾配降下法(SGD)で最適な行列(W)とバイアス(b)を学習
const learnedWeights = trainer.train({
  learningRate: 0.05,
  epochs: 200,
  regularization: 0.001
});

// 4. 学習した行列をアダプターに組み込む
adapter.addIntent("user_personalized_intent", learnedWeights);
```

### オンライン学習 (フィードバックループ)
ユーザーが検索結果をクリックするたびに、リアルタイムで行列を微調整（学習）することも可能です。

```typescript
// ユーザーがクリックした結果のベクトル (理想のベクトル)
const clickedVector = [...];

// 現在の意図行列を1ステップだけ更新（微調整）
const updatedWeights = trainer.updateOnline(
  currentWeights, // 現在の IntentWeights
  queryVector,    // 入力したクエリ
  clickedVector,  // クリックした対象
  0.01            // 学習率 (小さめに設定)
);

// 更新された意図を適用
adapter.addIntent("user_personalized_intent", updatedWeights);
```
これにより、システムを使えば使うほど、個人の意図（コンテキスト）に寄り添って検索空間が賢くなる体験を提供できます。

---

## 6. バイナリ・シリアライズ (超軽量保存と復元)

学習した巨大な意図行列をJSONとして保存するとファイルサイズが膨大になり、パース処理（`JSON.parse`）でメモリと時間を大量に消費します。`warpvector` は超軽量の **バイナリフォーマット (Uint8Array)** による高速なシリアライズをサポートしています。

### エクスポート (保存)
```typescript
// 学習済み、または定義済みの意図をバイナリとして抽出
const binaryData: Uint8Array = adapter.exportIntentBinary("user_personalized_intent");

// (Node.js/Bun 環境の場合、ファイルシステムへ保存)
import * as fs from "fs";
fs.writeFileSync("user_intent.wrpv", binaryData);
```

### インポート (復元)
```typescript
import { IntentAdapter } from "warpvector";

// (Node.js/Bun 環境の場合、ファイルから読み込み)
import * as fs from "fs";
const loadedBinary = fs.readFileSync("user_intent.wrpv");

// 次元数だけを指定して空の Adapter を作成
const adapter = new IntentAdapter(1536);

// バイナリデータをロードし、新しい意図として登録（JSONパース不要で超高速）
adapter.importIntentBinary("restored_intent", loadedBinary);

// すぐに推論に使用可能
const result = adapter.tune(queryVector, "restored_intent");
```
この機能は、エッジ環境（Cloudflare Workers など）にユーザーのパーソナライズデータを高速で読み込ませたい場合に絶大な威力を発揮します。
