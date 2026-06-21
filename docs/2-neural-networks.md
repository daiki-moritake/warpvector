# Neural Networks

WarpVector supports not only simple linear transformations (matrix multiplications) but also advanced spatial deformations using **Multi-Layer Perceptrons (MLP)** and **Non-linear Activation Functions**.

## 1. MlpAdapter

`MlpAdapter` takes the output from a pre-trained model and bends it intricately using multiple layers and non-linear functions, allowing for the segmentation of higher-order meanings.
Optimized for WASM (WebAssembly), it allows you to run blazing-fast neural network inference in browsers or edge environments without loading heavy machine learning frameworks like TensorFlow.js or PyTorch.

### Features
- **Arbitrary Layer Network Construction**: Define any combination of dimensions and activation functions from the input layer to the output layer.
- **Seamless Integration**: Has the same `tune()` interface as other `WarpAdapter`s, so it runs transparently as a plugin for Prisma or LangChain.
- **WASM-Driven**: The entire network executes completely within a single WASM module, eliminating delays caused by JS garbage collection.

### Basic Usage

```typescript
import { MlpAdapter } from 'warpvector';

// A 2-layer neural network that takes a 1536-dimensional input, passes through a 128-dimensional hidden layer, and outputs 2 dimensions (e.g., coordinates)
const mlp = new MlpAdapter([
  { inputDim: 1536, outputDim: 128, activation: "relu" },
  { inputDim: 128, outputDim: 2, activation: "linear" }
]);

// Set the weights for the 1st layer (1536 -> 128)
mlp.setLayerWeights(0, matrixLayer1, biasLayer1);

// Set the weights for the 2nd layer (128 -> 2)
mlp.setLayerWeights(1, matrixLayer2, biasLayer2);

const baseVector = /* 1536-dimensional vector obtained from OpenAI, etc. */;

// Ultra-fast non-linear inference (consistent processing inside WASM)
const outputVector = mlp.tune(baseVector); // Float32Array(2)
```

## 2. Non-linear Activation Functions

To express "spatial distortions" that cannot be fully captured by linear transformations alone, each core adapter (such as `IntentAdapter` and `MlpAdapter`) supports the application of non-linear activation functions after the transformation.

### Supported Functions

- **`relu`**: Sets negative values to 0 (sparsifies features)
- **`sigmoid`**: Smoothly squashes values into the 0.0 to 1.0 range
- **`tanh`**: Smoothly squashes values into the -1.0 to 1.0 range
- **`linear`** (default): No transformation applied

### Example: Using an Activation Function with IntentAdapter

Just specify the activation function as the third argument in the `tune` or `tuneBatch` methods, and it will be applied automatically. When using WASM, it is expanded and processed inline within WASM, meaning there is zero overhead.

```typescript
import { IntentAdapter } from 'warpvector';

const adapter = new IntentAdapter(myIntents);

// Cuts off negative noise components by passing them through the ReLU function
const activatedVector = adapter.tune(baseVector, "riskAnalysis", "relu");
```

### Why are Non-linear Transformations Necessary?

In simple searches, cosine similarity is common. However, when documents have sharp boundaries such as "positive vs. negative", a linear space might not be able to separate them cleanly.
By using MLPs or non-linear activation functions, you can bend the space to pull specific clusters apart, enabling advanced search tuning that "draws" only the intended search results closer.
