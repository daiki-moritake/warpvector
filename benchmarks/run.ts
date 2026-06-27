import { IntentAdapter, initWasm } from "@warpvector/core";
import { QuantizationAdapter } from "@warpvector/extras";

async function runBenchmark() {
  console.log("🚀 WarpVector Performance Benchmark");
  console.log("Initializing WASM...");
  await initWasm();

  const DIM = 1536;
  const NUM_VECTORS = 100_000;

  console.log(`Generating ${NUM_VECTORS} vectors of dimension ${DIM}...`);
  const vectors = new Float32Array(NUM_VECTORS * DIM);
  for (let i = 0; i < vectors.length; i++) {
    vectors[i] = Math.random() * 2 - 1; // [-1, 1]
  }

  // --- 1. Quantization Benchmark ---
  console.log("\n--- Quantization Benchmark ---");
  const int8Adapter = new QuantizationAdapter({ type: "int8", dim: DIM });
  const binaryAdapter = new QuantizationAdapter({ type: "binary", dim: DIM });

  console.log("Measuring Int8 Scalar Quantization...");
  const startInt8 = performance.now();
  for (let i = 0; i < NUM_VECTORS; i++) {
    int8Adapter.encode(vectors.subarray(i * DIM, (i + 1) * DIM));
  }
  const endInt8 = performance.now();
  console.log(`Int8 Quantization: ${((NUM_VECTORS / (endInt8 - startInt8)) * 1000).toFixed(0)} vecs/sec`);

  console.log("Measuring Binary Quantization...");
  const startBin = performance.now();
  for (let i = 0; i < NUM_VECTORS; i++) {
    binaryAdapter.encode(vectors.subarray(i * DIM, (i + 1) * DIM));
  }
  const endBin = performance.now();
  console.log(`Binary Quantization: ${(endBin - startBin).toFixed(2)} ms (${Math.round(NUM_VECTORS / ((endBin - startBin) / 1000))} vecs/sec)`);

  // --- 2. WASM Batch Transformation Benchmark ---
  console.log("\n--- Batch Transformation (Affine) Benchmark ---");
  
  // W: 1536 x 1536
  console.log("Generating 1536x1536 weight matrix...");
  const W = new Float32Array(DIM * DIM);
  for (let i = 0; i < W.length; i++) W[i] = Math.random();
  const b = new Float32Array(DIM);

  const intentAdapter = new IntentAdapter(DIM);
  intentAdapter.addIntent("bench", { matrix: W, bias: b });

  // バッチテストのために Flat な 2D 表現を使う
  // Float32Arrayの配列を作成
  const vectorList: Float32Array[] = [];
  for (let i = 0; i < 10000; i++) {
    vectorList.push(vectors.subarray(i * DIM, (i + 1) * DIM));
  }

  console.log(`Measuring WASM batch transformation for ${vectorList.length} vectors...`);
  const startWasm = performance.now();
  intentAdapter.tuneBatch(vectorList, "bench");
  const endWasm = performance.now();
  console.log(`WASM tuneBatch: ${(endWasm - startWasm).toFixed(2)} ms (${Math.round(vectorList.length / ((endWasm - startWasm) / 1000))} vecs/sec)`);
  
  console.log("\nBenchmark Complete!");
}

runBenchmark().catch(console.error);
