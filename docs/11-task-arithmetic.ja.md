
# タスクベクトル演算 (Task Arithmetic)

## 概要

Task Arithmetic は、複数の学習済みアダプタの重み（`IntentWeights`）を「タスクベクトル」として加減算し、新しい静的アダプタ行列を推論時ゼロオーバーヘッドで合成する手法です。

数式:
$$W_{\text{new}} = W_{\text{base}} + \sum_i \text{scale}_i \cdot (W_i - W_{\text{base}})$$

これにより、推論時に複数のアダプタを順番に適用する必要なく、単一のマージ済み行列で即座に変換できます。

## ユースケース

1. **マルチドメイン検索**: 法律・金融・医療など個別に学習した重みを、指定の割合でマージ
2. **A/B テスト**: 異なるドメイン適応の重みを混ぜ合わせて検索品質を評価
3. **負のスケール**: `scale: -0.5` で特定ドメインの影響を逆方向に打ち消す

## 使い方

### 基本的なマージ

```typescript
import { TaskArithmetic } from '@warpvector/extras';

// 「法律ドメイン」と「金融ドメイン」の学習済み重みをマージ
const mergedWeights = TaskArithmetic.merge([
  { weights: legalWeights, scale: 0.7 },   // 法律を70%
  { weights: financeWeights, scale: 0.3 },  // 金融を30%
]);

// マージされた重みは通常の IntentWeights として即座に使用可能
adapter.addIntent("legal_finance", mergedWeights);
```

### ベース意図を指定したマージ

```typescript
// カスタムのベース意図を基準に、差分を合成
const mergedWeights = TaskArithmetic.merge(
  [
    { weights: specializedWeights1, scale: 0.5 },
    { weights: specializedWeights2, scale: 0.5 },
  ],
  baseWeights // ベースとなる意図の重み (省略時は恒等行列)
);
```

### 負のスケール（タスクの打ち消し）

```typescript
// 「雑音」タスクの影響を反転させて打ち消す
const denoisedWeights = TaskArithmetic.merge([
  { weights: goodTaskWeights, scale: 1.0 },
  { weights: noiseTaskWeights, scale: -0.3 }, // マイナススケールで打ち消し
]);
```

## API

### `TaskArithmetic.merge(tasks, baseIntent?)`

| 引数 | 型 | 説明 |
|---|---|---|
| `tasks` | `TaskConfig[]` | 合成するタスクのリスト |
| `baseIntent` | `IntentWeights?` | 基準となる重み。省略時は恒等行列 + ゼロバイアス |

#### `TaskConfig`

| フィールド | 型 | 説明 |
|---|---|---|
| `weights` | `IntentWeights` | 学習済みの重み |
| `scale` | `number` | スケール係数。1.0 が標準、マイナスで逆効果 |
