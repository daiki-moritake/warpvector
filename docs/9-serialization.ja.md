# 状態の永続化・シリアライズ (Universal Serialization)

WarpVector の多くのアダプタ（特に `WhiteningAdapter` によるPCA学習結果や、`Trainer` によって最適化された `IntentAdapter` / `MlpAdapter` の重み）は、実行時に「状態」を持ちます。

Cloudflare Workers や Vercel Edge Functions などのサーバーレス/エッジ環境は揮発性（ステートレス）であるため、リクエストが終了するとメモリ上の状態は消えてしまいます。
そのため、WarpVector は **全アダプタの状態を安全かつ即座に JSON 形式でシリアライズ / デシリアライズできる Universal Serialization 機能** を提供しています。

## サポートされているアダプタ

現在、以下のすべてのアダプタで `exportState()` と静的メソッド `importState()` がサポートされています。

- `IntentAdapter` / `LoraIntentAdapter`
- `WhiteningAdapter`
- `ProjectionAdapter`
- `MlpAdapter`

## 基本的な使い方

### 保存 (Export)

ストリーミング学習やデータの更新が終わったタイミングで `exportState()` を呼び出し、得られた文字列（JSON形式）を Redis, Cloudflare KV, S3, または RDB のテキストカラムなどに保存します。

```typescript
import { WhiteningAdapter } from 'warpvector';

const adapter = new WhiteningAdapter(1536);

// ... ベクトルを受信してオンライン学習 (update) を複数回実行 ...
adapter.update(vectorA);
adapter.update(vectorB);

// 学習後の状態（主成分ベクトル等）をシリアライズ
const stateString = adapter.exportState(); 
// -> '{"dimension":1536,"learningRate":0.001, ... "components":[...]}'

// 例: Redis などに保存
await redis.set("my_whitening_state", stateString);
```

### 復元 (Import)

次回起動時や別インスタンスで処理を行う際、保存しておいた文字列を `importState()`（クラスの静的メソッド）に渡すだけで、全く同じ状態のアダプタが即座に復元されます。

```typescript
import { WhiteningAdapter } from 'warpvector';

// 例: Redis 等から状態を読み込む
const stateString = await redis.get("my_whitening_state");

// インスタンスを復元
const restoredAdapter = WhiteningAdapter.importState(stateString);

// 以前の学習結果を引き継いだ状態で、すぐに検索のチューニングや追加学習が可能
const whitenedVector = restoredAdapter.tune(newVector);
```

## MlpAdapter の復元に関する注意点

`MlpAdapter` はバックエンドに WebAssembly (WASM) を使用しているため、復元後に WASM 側のメモリとインスタンスを再構築する必要があります。
そのため、`MlpAdapter.importState()` で復元した直後に **必ず `await mlp.init()` を呼び出す** 必要があります（これを行わないとエラーになります）。

```typescript
import { MlpAdapter } from 'warpvector';

const stateString = await redis.get("my_mlp_state");

// 1. JSオブジェクトとしての状態を復元
const mlp = MlpAdapter.importState(stateString);

// 2. WASM インスタンスを再初期化 (非同期)
await mlp.init();

// 3. 推論可能になる
const result = mlp.tune(vector);
```

## Binary Serialization (バイナリ保存) について

`IntentAdapter` には JSON によるシリアライズの他に、`Uint8Array` を直接出力する `exportIntentBinary()` / `importIntentBinary()` メソッドも備わっています。
JSON パースのオーバーヘッドさえも削減したい極限のパフォーマンスが求められる環境（例: IndexedDB への保存や WebGL テクスチャへのマッピング前）では、こちらのバイナリシリアライゼーションを使用することも可能です。
