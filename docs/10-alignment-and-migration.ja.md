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

---

## AlignmentAdapter & MigrationTrainer — ゼロダウンタイムでのモデル移行

埋め込みモデルを切り替える際（例: `text-embedding-ada-002` (1536次元) → `text-embedding-3-small` (512次元)）、通常はDB内の数百万件のベクトルを全て新しいモデルで計算し直す（再インデックス）必要があり、莫大なコストとダウンタイムが生じます。

`AlignmentAdapter` を使えば、新しいモデルのクエリベクトルを**瞬時に古いモデルのベクトル空間へ翻訳（アラインメント）**できるため、再インデックスを一切行うことなくモデルの移行が可能になります（ベンダーロックインの解消）。

### ワークフロー

1. 同一テキスト（100〜500件程度）を新旧両方のモデルで埋め込む
2. ペアデータとして `MigrationTrainer` に追加
3. Adam オプティマイザーで射影行列を学習
4. 学習済みの `ProjectionWeights` を `AlignmentAdapter` に渡して実行

### 使い方

```typescript
import { MigrationTrainer } from "@warpvector/train";
import { AlignmentAdapter } from '@warpvector/core';

// 新モデル (512次元) のクエリを 旧モデル (1536次元) の空間へ翻訳する学習
const trainer = new MigrationTrainer(512, 1536);

// 同一テキストの埋め込みペアを追加
trainer.addExample({ source: newSmallVec1, target: oldAdaVec1 });
trainer.addExample({ source: newSmallVec2, target: oldAdaVec2 });
trainer.addExample({ source: newSmallVec3, target: oldAdaVec3 });

// 学習 (autoTune で最適学習率を自動探索)
const alignmentWeights = await trainer.train({ epochs: 200, autoTune: true });

// 学習済み射影行列を AlignmentAdapter に適用
const migrator = new AlignmentAdapter(512, 1536, {
  migration: alignmentWeights,
});

// 新しいクエリベクトルを旧DBの空間に翻訳！
const translatedVector = migrator.align(newQueryVector, "migration");

// この translatedVector を使って、そのまま古いDBを検索できます！
```

### autoTune (最適学習率探索)

`train({ autoTune: true })` を指定すると、内部で候補の学習率を短時間テストし、最も損失が小さくなる値を自動選択します。
