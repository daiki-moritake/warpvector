# Task Vector Arithmetic (Task Arithmetic)

## Overview

Task Arithmetic is a technique that adds or subtracts the weights (`IntentWeights`) of multiple pre-trained adapters as "task vectors" to synthesize a new, static adapter matrix with zero overhead at inference time.

Formula:
$$W_{\text{new}} = W_{\text{base}} + \sum_i \text{scale}_i \cdot (W_i - W_{\text{base}})$$

This allows you to instantly transform using a single merged matrix, rather than needing to sequentially apply multiple adapters during inference.

## Use Cases

1. **Multi-domain Search**: Merge separately learned weights for law, finance, medicine, etc., at specified ratios.
2. **A/B Testing**: Mix weights from different domain adaptations to evaluate search quality.
3. **Negative Scaling**: Cancel out the influence of a specific domain in the opposite direction using `scale: -0.5`.

## Usage

### Basic Merging

```typescript
import { TaskArithmetic } from '@warpvector/extras';

// Merge pre-trained weights from the "Legal Domain" and "Finance Domain"
const mergedWeights = TaskArithmetic.merge([
  { weights: legalWeights, scale: 0.7 },   // 70% Legal
  { weights: financeWeights, scale: 0.3 },  // 30% Finance
]);

// The merged weights can be used immediately as normal IntentWeights
adapter.addIntent("legal_finance", mergedWeights);
```

### Merging with a Specified Base Intent

```typescript
// Synthesize the differences relative to a custom base intent
const mergedWeights = TaskArithmetic.merge(
  [
    { weights: specializedWeights1, scale: 0.5 },
    { weights: specializedWeights2, scale: 0.5 },
  ],
  baseWeights // The weights of the base intent (defaults to the identity matrix if omitted)
);
```

### Negative Scaling (Task Cancellation)

```typescript
// Invert and cancel out the influence of a "noise" task
const denoisedWeights = TaskArithmetic.merge([
  { weights: goodTaskWeights, scale: 1.0 },
  { weights: noiseTaskWeights, scale: -0.3 }, // Cancel out with a negative scale
]);
```

## API

### `TaskArithmetic.merge(tasks, baseIntent?)`

| Argument | Type | Description |
|---|---|---|
| `tasks` | `TaskConfig[]` | List of tasks to synthesize |
| `baseIntent` | `IntentWeights?` | Baseline weights. If omitted, uses the identity matrix + zero bias |

#### `TaskConfig`

| Field | Type | Description |
|---|---|---|
| `weights` | `IntentWeights` | Pre-trained weights |
| `scale` | `number` | Scaling factor. 1.0 is standard, negative values have the opposite effect |
