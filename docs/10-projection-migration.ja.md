
# 次元削減・モデル間移行 (Projection & Migration)

## ProjectionAdapter — 射影変換による次元削減・拡張

PCA や SVD で事前計算した射影行列を用いて、ベクトルの次元数を変換します。内部では WASM による高速な行列ベクトル積を実行し、純粋な JS 実装への自動フォールバックも備えています。

### 典型的なユースケース

1. **ベクトルの圧縮**: 1536次元の埋め込みを 512 や 256 次元に削減し、メモリと検索コストを削減
2. **特徴量抽出**: PCA で得られた主成分空間への射影
3. **パイプライン内での次元変換**: `WarpPipeline` の `addProjection()` でチェーンに組み込み可能

### 使い方

```typescript
import { ProjectionAdapter } from '@warpvector/core';

// 1536次元 → 512次元の射影行列を設定
const adapter = new ProjectionAdapter(1536, 512, {
  pca: { matrix: pcaMatrix, bias: pcaBias }
});

// 次元削減を実行 (WASM が利用可能なら自動使用)
const compressed = adapter.tune(baseVector, "pca"); // 512次元の Float32Array
```

### パイプラインでの使用

```typescript
import { WarpPipeline } from '@warpvector/core';

const pipeline = new WarpPipeline(1536)
  .addIntent(intentWeights)       // まず意図による空間変形
  .addProjection(512, { v1: projWeights }); // その後次元削減

// inputDim は自動的に 512 に更新されます
```

### 状態の永続化

```typescript
// エクスポート
const state = adapter.exportState(); // JSON 文字列

// インポート
const restored = ProjectionAdapter.importState(state);
```

---

## MigrationTrainer — モデル間ベクトル空間の翻訳

埋め込みモデルを切り替える際（例: `text-embedding-ada-002` → `text-embedding-3-small`）、既存のベクトルを新モデルの空間に翻訳する射影行列を自動学習します。

これにより、全データを新モデルで再埋め込みする必要なく、旧ベクトルを新空間に近似的にマッピングできます。

### ワークフロー

1. 同一テキストを新旧両方のモデルで埋め込む
2. ペアデータとして `MigrationTrainer` に追加
3. Adam オプティマイザーで射影行列を学習
4. 学習済みの `ProjectionWeights` を `ProjectionAdapter` に渡して実行

### 使い方

```typescript
import { MigrationTrainer } from "@warpvector/train";
import { ProjectionAdapter } from '@warpvector/core';

// 1536次元 (ada-002) → 512次元 (3-small) の射影行列を学習
const trainer = new MigrationTrainer(1536, 512);

// 同一テキストの埋め込みペアを追加
trainer.addExample({ source: adaVec1, target: smallVec1 });
trainer.addExample({ source: adaVec2, target: smallVec2 });
trainer.addExample({ source: adaVec3, target: smallVec3 });

// 学習 (autoTune で最適学習率を自動探索)
const projWeights = await trainer.train({ epochs: 200, autoTune: true });

// 学習済み射影行列を ProjectionAdapter に適用
const migrator = new ProjectionAdapter(1536, 512, {
  migration: projWeights,
});

// 旧ベクトルを新空間に変換
const newSpaceVector = migrator.tune(oldAdaVector, "migration");
```

### autoTune (最適学習率探索)

`train({ autoTune: true })` を指定すると、内部で5つの候補学習率（0.1, 0.05, 0.01, 0.005, 0.001）を短時間テストし、最も損失が小さくなる値を自動選択します。
