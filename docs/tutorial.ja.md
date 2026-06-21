# WarpVector E2E チュートリアル: ゼロから意図行列を作る

このチュートリアルでは、WarpVector を使って **「検索クエリのベクトルを、特定の意図に合わせて空間変換する」** という実践的なシナリオを最初から最後まで構築します。

## 目次

1. [コンセプト: 意図行列とは何か](#コンセプト-意図行列とは何か)
2. [セットアップ](#セットアップ)
3. [Step 1: IntentAdapter で線形変換を構築](#step-1-intentadapter-で線形変換を構築)
4. [Step 2: パイプラインの組み立て](#step-2-パイプラインの組み立て)
5. [Step 3: 量子化で保存コストを削減](#step-3-量子化で保存コストを削減)
6. [Step 4: 状態の保存と復元](#step-4-状態の保存と復元)
7. [Step 5: MLP アダプタで非線形変換](#step-5-mlp-アダプタで非線形変換)
8. [API リファレンス](#api-リファレンス)

---

## コンセプト: 意図行列とは何か

ベクトル検索において、同じクエリでも **「類似度重視」** と **「多様性重視」** では
最適なベクトル空間が異なります。WarpVector の **意図行列（Intent Matrix）** は、
この「意図に応じたベクトル空間の回転・拡縮」を実現します。

```
元のベクトル空間        意図行列 W        変換後の空間
     [v₁]           × [W_intent]   →    [v₁']
     [v₂]                               [v₂']
```

数学的には `v' = Wv + b` （アフィン変換）です。

---

## セットアップ

```bash
# 新しいプロジェクトを作成
mkdir my-warp-project && cd my-warp-project
bun init -y

# WarpVector をインストール
bun add @warpvector/core
```

---

## Step 1: IntentAdapter で線形変換を構築

最も基本的な使い方は、`IntentAdapter` に意図行列を登録し、ベクトルを変換することです。

```typescript
import { IntentAdapter } from "@warpvector/core";

// 3次元ベクトルを扱う IntentAdapter を作成
const adapter = new IntentAdapter(3);

// 「類似度重視」の意図行列を登録
// この行列は、第1次元（意味的類似度）を2倍に強調し、
// 第3次元（多様性スコア）を半分に抑える変換を表します。
adapter.addIntent("similarity", {
  matrix: [
    [2, 0, 0],   // 第1次元を2倍に強調
    [0, 1, 0],   // 第2次元はそのまま
    [0, 0, 0.5], // 第3次元を半分に
  ],
  bias: [0, 0, 0], // バイアスなし
});

// 「多様性重視」の意図行列
adapter.addIntent("diversity", {
  matrix: [
    [0.5, 0, 0], // 第1次元を半分に
    [0, 1, 0],
    [0, 0, 2],   // 第3次元を2倍に強調
  ],
  bias: [0, 0, 0],
});

// ベクトルを変換
const query = [0.8, 0.5, 0.3];
const forSimilarity = adapter.tune(query, "similarity");
// → Float32Array [1.6, 0.5, 0.15]

const forDiversity = adapter.tune(query, "diversity");
// → Float32Array [0.4, 0.5, 0.6]
```

---

## Step 2: パイプラインの組み立て

複数の変換を **パイプライン** として直列に接続できます。

```typescript
import { WarpPipeline } from "@warpvector/core";

// パイプライン: 意図変換 → 次元削減(3D → 2D)
const pipeline = new WarpPipeline(3)
  .addIntent({
    search: {
      matrix: [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      bias: [0.1, 0, 0],
    },
  })
  .addProjection(2, {
    search: {
      matrix: [
        [1, 0, 0], // 第1・第2次元だけを取り出す
        [0, 1, 0],
      ],
    },
  });

// 実行
const result = pipeline.run([0.5, 0.8, 0.3], { intent: "search" });
// → Float32Array [1.1, 0.8]  (3次元 → 2次元に圧縮)
```

---

## Step 3: 量子化で保存コストを削減

`@warpvector/extras` の `QuantizationAdapter` を使い、
Float32 ベクトルを Int8 に量子化してストレージコストを **1/4** に削減できます。

```typescript
import { WarpPipeline } from "@warpvector/core";
import { QuantizationAdapter } from "@warpvector/extras";

const quantizer = new QuantizationAdapter({ type: "int8", dim: 2 });

// パイプラインの最終段に量子化を設定（FinalStageAdapter パターン）
const pipeline = new WarpPipeline(3)
  .addIntent({
    search: {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      bias: [0, 0, 0],
    },
  })
  .addProjection(2, {
    search: {
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
      ],
    },
  })
  .setFinalStage("QuantizationAdapter", quantizer);

const result = pipeline.run([0.5, 0.8, 0.3], { intent: "search" });
// → Int8Array [64, 102]  (Float32Array → Int8Array に量子化)
```

> **Note:** `setFinalStage()` は従来の `addStep("QuantizationAdapter", ...)` とは異なり、
> パイプラインの中間段には Float32Array のみを強制するため型安全です。

---

## Step 4: 状態の保存と復元

パイプラインの重み（意図行列、射影行列など）はすべて JSON にシリアライズ・復元可能です。

```typescript
// 保存
const state = pipeline.exportState();
const json = JSON.stringify(state);
// → データベースやファイルに保存

// 復元
const restored = WarpPipeline.importState(JSON.parse(json));
const result2 = restored.run([0.5, 0.8, 0.3], { intent: "search" });
// → 保存前と完全に同じ結果
```

---

## Step 5: MLP アダプタで非線形変換

線形変換では表現できない複雑な空間変換には、`MlpAdapter`（多層パーセプトロン）を使います。
WASM による高速推論で、1536次元のベクトルも数マイクロ秒で処理できます。

```typescript
import { MlpAdapter } from "@warpvector/ml";

const mlp = new MlpAdapter([
  {
    // Layer 1: 3 → 4 (ReLU)
    matrix: [
      [0.5, 0.3, 0.1],
      [0.2, 0.8, 0.4],
      [0.1, 0.1, 0.9],
      [0.4, 0.2, 0.3],
    ],
    bias: [0.1, 0, 0, 0.05],
    activation: "relu",
  },
  {
    // Layer 2: 4 → 2 (Linear)
    matrix: [
      [1, 0, 0.5, 0.3],
      [0, 1, 0.2, 0.8],
    ],
    bias: [0, 0],
    activation: "linear",
  },
]);

// WASM の初期化（1回だけ）
await mlp.init();

// 推論
const output = mlp.tune([0.5, 0.8, 0.3]);
// → Float32Array (2次元に次元削減された非線形変換結果)

// パイプラインに組み込む
const pipeline = new WarpPipeline(3)
  .addStep("MlpAdapter", mlp);

await pipeline.init(); // WASM アダプタの初期化
const pipelineResult = pipeline.run([0.5, 0.8, 0.3]);
```

---

## API リファレンス

| クラス | 用途 | パッケージ |
|--------|------|-----------|
| `IntentAdapter` | 意図ごとの線形変換 (Wx+b) | `@warpvector/core` |
| `LoraIntentAdapter` | LoRA式の低ランク意図変換 | `@warpvector/core` |
| `ProjectionAdapter` | PCA/SVD による次元削減 | `@warpvector/core` |
| `WarpPipeline` | アダプタの直列接続 | `@warpvector/core` |
| `QuantizationAdapter` | Int8/Binary 量子化 | `@warpvector/extras` |
| `ColbertAdapter` | Late Interaction (MaxSim) | `@warpvector/extras` |
| `MlpAdapter` | 非線形変換 (WASM MLP) | `@warpvector/ml` |
| `WhiteningAdapter` | Whitening 正規化 | `@warpvector/ml` |

### パイプラインの設計パターン

```
入力ベクトル (Float32Array)
    │
    ├─ WarpAdapter (IntentAdapter, ProjectionAdapter, MlpAdapter...)
    │   → 常に Float32Array を返す
    │   → 複数を直列に接続可能
    │
    └─ FinalStageAdapter (QuantizationAdapter)
        → Int8Array / Uint8Array を返す
        → パイプライン最終段にのみ配置可能
        → setFinalStage() で設定
```
