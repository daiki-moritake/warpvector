# Quantization and Compression

High-dimensional vectors (e.g., 1536-dimensional `Float32`) consume 6,144 bytes (about 6KB) of memory per vector. Trying to hold and search 1 million vectors in-memory would require about 6GB of memory just for that, which is completely unmanageable in browsers or edge environments.

WarpVector's `QuantizationAdapter` provides the ability to drastically compress (quantize) the data size while preserving the semantic information of the vector as much as possible.

## 1. Int8 Scalar Quantization (1/4 Memory)

This is a technique that scales and rounds the value of each dimension of a 32-bit floating-point (Float32) into an 8-bit integer (Int8) ranging from -128 to 127.
The accuracy degradation is extremely small (usually less than 1-2% score error), while strictly reducing the memory usage to **one-quarter**.

### Usage

```typescript
import { QuantizationAdapter } from 'warpvector';

// Create an Int8 quantization adapter
const int8Adapter = new QuantizationAdapter({ type: "int8", dim: 1536 });

// A 1536-dimensional Float32Array
const floatVector = getFloat32Vector();

// Execute quantization (Returns an Int8Array)
const int8Vec = int8Adapter.tune(floatVector);

// Score calculation during search (Dot Product)
// Uses a fast dot product specifically for Int8 as an approximation of Float32 cosine similarity
const similarityScore = QuantizationAdapter.int8DotProduct(int8Vec1, int8Vec2);
```

## 2. Binary (1-bit) Quantization (1/32 Memory)

This technique compresses values down to the extreme limit of a "bit" that is 0 or 1. If a value is greater than 0, it is judged as 1; if less, as 0.
Furthermore, because it "packs" 32 bits into a single number (a 32-bit integer), the memory usage becomes an astonishing **1/32** (just 192 bytes for 1536 dimensions).

### Usage and Hamming Distance

To calculate the distance between Binary quantized vectors, you use "Hamming Distance" instead of cosine similarity. This counts the "number of differing bits" in each other's bit strings. Since it can be calculated with CPU-level XOR operations and bit counting (Popcount), **even searching through tens of millions of records completes instantly**.

```typescript
import { QuantizationAdapter } from 'warpvector';

// Create a Binary quantization adapter
const binaryAdapter = new QuantizationAdapter({ type: "binary", dim: 1536 });

// Execute quantization (Returns data packed as a Uint8Array)
// 1536 dimensions -> Compressed into 192 Uint8s
const binVec = binaryAdapter.tune(floatVector);

// Distance calculation during search (Hamming Distance)
// The *smaller* the distance, the higher the similarity
const distance = QuantizationAdapter.hammingDistance(binVec1, binVec2);

// Application: If you want to convert the distance (0-1536) into similarity (0-1)
const similarity = 1.0 - (distance / 1536);
```

## 3. When to use which?

- **Int8 Quantization**: When you want to minimize the loss of search accuracy (Ranking) as much as possible. It is extremely powerful for optimizing the memory of the main VectorDB.
- **Binary Quantization**: When you want to perform "candidate narrowing (Retrieval)" at ultra-high speed and with very low memory. The best practice is a "Two-stage Retrieval" architecture: use Binary search to roughly and incredibly quickly pick up 1,000 candidates out of 1 million data points, and then use normal Float32 or Int8 to re-sort (Rerank) them with accurate scores.
