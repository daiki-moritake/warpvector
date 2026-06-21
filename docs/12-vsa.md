# Hyperdimensional Computing / VSA (Vector Symbolic Architecture)

## Overview

`VsaAdapter` provides operations for Vector Symbolic Architecture (VSA) / Hyperdimensional Computing.

By logically and mathematically combining (binding) or aggregating (bundling) vectors together, you can embed keys and values (like metadata) into a single dense vector, allowing you to perform operations directly on the search space.

## 3 Basic Operations

### 1. Bundle (Superposition)

Adds (superposes) multiple vectors together to integrate them into a single vector. Used when creating a vector that "contains both concepts A and B".

```typescript
import { VsaAdapter } from '@warpvector/extras';

// Integrate the concepts of "Science" and "Technology"
const sciTech = VsaAdapter.bundle([scienceVec, technologyVec]);
// sciTech has high cosine similarity to both
```

### 2. Bind (Hadamard Product)

"Combines" two vectors using the Hadamard product (element-wise product). By multiplying a key (User ID) and a value (Preference), it generates a unique vector.

```typescript
// Combine a User ID vector and a Preference vector
const bound = VsaAdapter.bind(userIdVec, preferenceVec);
```

### 3. Unbind

Extracts the original value from a bound vector using one of the keys.

```typescript
// Extract the Preference vector using the User ID vector as the key
const recovered = VsaAdapter.unbind(bound, userIdVec);
// recovered ≈ preferenceVec (Approximately recovered)
```

## Binary VSA (XOR Operations)

These are ultra-fast VSA operations for `Uint8Array` vectors that have been 1-bit (Binary) quantized using `QuantizationAdapter`. By using XOR operations, extremely high-speed processing with minimal memory is possible.

### bindBinary / unbindBinary

Performs binding and unbinding utilizing the self-inverse property of XOR (`A ^ B ^ B = A`).

```typescript
const binaryBound = VsaAdapter.bindBinary(binKey, binValue);
const binaryRecovered = VsaAdapter.unbindBinary(binaryBound, binKey);
// binaryRecovered === binValue
```

### bundleBinary (Majority Vote)

Superposes multiple binary vectors. Determines the final bit through a Majority Vote of 1s and 0s at each bit position.

```typescript
const merged = VsaAdapter.bundleBinary([bin1, bin2, bin3]);
```

## Use Cases

### Embedded Metadata Search

An example of embedding metadata into vectors to perform search and attribute filtering simultaneously:

```typescript
// 1. Bind each attribute with a key vector
const categoryBound = VsaAdapter.bind(categoryKeyVec, categoryValueVec);
const priceBound = VsaAdapter.bind(priceKeyVec, priceRangeVec);

// 2. Bundle the original vector and the metadata
const enrichedDoc = VsaAdapter.bundle([
  documentVec,
  categoryBound,
  priceBound,
]);

// 3. Extract specific metadata during search
const extractedCategory = VsaAdapter.unbind(enrichedDoc, categoryKeyVec);
```

## API

### Dense Vector Operations

| Method | Description |
|---|---|
| `VsaAdapter.bundle(vectors, options?)` | Superpose multiple vectors (with L2 normalization) |
| `VsaAdapter.bind(vec1, vec2, options?)` | Bind via Hadamard Product |
| `VsaAdapter.unbind(boundVec, keyVec, options?)` | Unbind via element-wise division |

### Binary Vector Operations

| Method | Description |
|---|---|
| `VsaAdapter.bindBinary(bin1, bin2)` | Binary bind via XOR |
| `VsaAdapter.unbindBinary(boundBin, keyBin)` | Extract via XOR's self-inverse property |
| `VsaAdapter.bundleBinary(bins)` | Binary bundle via Majority Vote |

### VsaOptions

| Field | Type | Default | Description |
|---|---|---|---|
| `shouldNormalize` | `boolean` | `true` | Whether to L2 normalize the result |
