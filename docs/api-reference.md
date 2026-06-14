# API Reference

`warpvector` で提供される主要なクラスと関数のリファレンスです。

---

## Classes

### `WarpPipeline`
複数のアダプタを直感的に繋ぎ合わせ、ベクトルの変換、非同期初期化、バッチ処理、DBフォーマット出力を一括で管理できる統合インターフェース。

- `constructor(inputDim: number)`
- `addIntent(intents?: Record<string, IntentWeights>): this`
- `addLoraIntent(rank: number, intents?: Record<string, LoraIntentWeights>): this`
- `addWhitening(options?: { learningRate?: number; numComponents?: number }): this`
- `addProjection(outputDim: number, projections?: Record<string, ProjectionWeights>): this`
- `addMlp(layers: MlpLayer[]): this`
- `quantize(type: QuantizationType): this`
- `addStep(type: string, adapter: WarpAdapter): this`
  - カスタムアダプタを直接パイプラインの末尾に追加します。
- `init(): Promise<void>`
  - WASM等の非同期初期化が必要な組み込みアダプタを一括でセットアップします。
- `run(vector: number[] | Float32Array, context?: RunContext): any`
  - 構成された全ての変換ステップを順次適用します。
- `runBatch(vectors: (number[] | Float32Array)[], context?: RunContext): any[]`
  - 複数のベクトルを一括でパイプラインに流し込みます。WASM対応アダプタではバッチ処理が並列化されます。
- `runAndFormat(vector: number[] | Float32Array, dbOptions: { format: string, topK?: number, filter?: any }, context?: RunContext): any`
  - 変換からデータベース向けフォーマット（pinecone, pgvector, redis）までの処理を1行で行います。
- `exportState(): PipelineState[]`
- `static importState(states: PipelineState[]): WarpPipeline`

### `IntentAdapter`
インメモリでベクトルのアフィン変換を行うメインクラス。

- `constructor(intents: Record<string, IntentWeights>)`
  - 初期化時に指定された意図をロードし、WASM/Float32Array向けに最適化します。
- `tune(baseVector: number[] | Float32Array, intent: string, activation?: Activation): Float32Array`
  - 指定した意図のアフィン変換を単一のベクトルに適用します。
- `tuneBatch(baseVectors: (number[] | Float32Array)[], intent: string, activation?: Activation): Float32Array[]`
  - 複数のベクトルに一括で変換を適用します。可能であれば WASM / SIMD により高速化されます。
- `tuneBlended(baseVector: number[] | Float32Array, blendWeights: Record<string, number>, activation?: Activation): Float32Array`
  - 複数の意図を指定した割合（例: `{ intentA: 0.7, intentB: 0.3 }`）で合成し、適用します。
- `tuneBatchBlended(...)`
  - `tuneBlended` のバッチ処理（WASM対応）版。
- `tuneAutoBlended(baseVector: number[] | Float32Array, activation?: Activation): Float32Array`
  - `routingVector`（代表ベクトル）の設定に基づき、クエリベクトル自身から最適なブレンド割合を自動で計算して適用します。
- `addIntent(intentName: string, weights: IntentWeights): void`
- `removeIntent(intentName: string): void`

### `LoraIntentAdapter`
高次元（1536次元など）のベクトル向けに、低ランク行列(LoRA)を用いてメモリと計算量を劇的に削減するアダプター。

- `constructor(dimension: number, rank: number, intents?: Record<string, LoraIntentWeights>)`
- `tune(baseVector: number[] | Float32Array, intent: string): Float32Array`

### `ProjectionAdapter`
PCAやSVDなどで計算された射影行列を用いて、次元削減や次元拡張を行うためのアダプター。

- `constructor(inDimension: number, outDimension: number, projections?: Record<string, ProjectionWeights>)`
- `project(baseVector: number[] | Float32Array, projectionName: string): Float32Array`

---

## Utility Functions (`utils.ts`)

- `normalize(vector: number[] | Float32Array): Float32Array`
  - ベクトルのL2ノルムを計算し、長さを1に正規化します。
- `cosineSimilarity(v1: number[] | Float32Array, v2: number[] | Float32Array): number`
  - 2つのベクトルのコサイン類似度（-1.0〜1.0）を計算します。
- `innerProduct(v1: number[] | Float32Array, v2: number[] | Float32Array): number`
  - 2つのベクトルの内積を計算します。
- `slerp(v1: number[] | Float32Array, v2: number[] | Float32Array, t: number): Float32Array`
  - 球面線形補間。コサイン類似度の構造を維持したまま2つのベクトル間を補間します（t は 0.0〜1.0）。
- `reject(baseVector: number[] | Float32Array, negativeVector: number[] | Float32Array): Float32Array`
  - 直交射影による成分除去。`baseVector` から `negativeVector` の成分を完全に消し去ります。
- `applyActivationToVector(vector: Float32Array, activation?: "relu" | "sigmoid" | "tanh"): void`
  - ベクトルに対してインプレースで非線形活性化関数を適用します。
- `softmax(values: number[]): number[]`
  - 数値配列を確率分布（合計1.0）に変換します。オーバーフロー防止機構付き。

---

## Types

- `IntentWeights`: `{ matrix: number[][], bias: number[], routingVector?: number[] }`
- `LoraIntentWeights`: `{ matrixA: number[][], matrixB: number[][], bias: number[] }`
- `ProjectionWeights`: `{ matrix: number[][] }`
- `Activation`: `"relu" | "sigmoid" | "tanh"`
