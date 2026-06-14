import { MlpAdapter, MlpLayer } from "../src/MlpAdapter";

/**
 * 【WASMを用いた超軽量・非線形MLP推論デモ】
 * 
 * WarpVectorは単なるアフィン変換（1層の行列積）だけでなく、
 * ReLUやSigmoidを含む多層ニューラルネットワーク（MLP）の推論を
 * WASMを用いて超高速に実行することができます。
 * 
 * 外部の重厚なTensorFlow.jsやONNX.jsなどに依存せず、
 * エッジデバイスやブラウザ内で完結するゼロ依存の推論エンジンとして機能します。
 */
async function runMlpDemo() {
  console.log("=== WASM MLP (多層パーセプトロン) 超高速推論デモ ===\n");

  // 例: 3次元入力 -> (ReLU) -> 128次元隠れ層 -> (Sigmoid) -> 2次元出力 のネットワーク構造
  const inputDim = 3;
  const hiddenDim = 128;
  const outputDim = 2;

  // ランダムな重みでネットワークを初期化 (本来は学習済みの重みをロードします)
  const layer1Matrix: number[][] = [];
  const layer1Bias: number[] = [];
  for (let r = 0; r < hiddenDim; r++) {
    const row = [];
    for (let c = 0; c < inputDim; c++) {
      row.push(Math.random() * 2 - 1);
    }
    layer1Matrix.push(row);
    layer1Bias.push(Math.random() * 2 - 1);
  }

  const layer2Matrix: number[][] = [];
  const layer2Bias: number[] = [];
  for (let r = 0; r < outputDim; r++) {
    const row = [];
    for (let c = 0; c < hiddenDim; c++) {
      row.push(Math.random() * 2 - 1);
    }
    layer2Matrix.push(row);
    layer2Bias.push(Math.random() * 2 - 1);
  }

  const layers: MlpLayer[] = [
    {
      matrix: layer1Matrix,
      bias: layer1Bias,
      activation: "relu" // 第1層の出力はReLUを通す
    },
    {
      matrix: layer2Matrix,
      bias: layer2Bias,
      activation: "sigmoid" // 第2層(出力層)の出力はSigmoidを通す
    }
  ];

  console.log("🚀 MlpAdapter を初期化中 (WASMコンパイルとメモリ確保)...");
  const adapter = new MlpAdapter(layers);
  
  // 必須: WASMの非同期初期化
  await adapter.init();
  console.log("✅ 初期化完了\n");

  const testVector = [0.5, -0.2, 0.9];
  console.log(`📥 入力ベクトル (${inputDim}次元):`, testVector);

  // ウォームアップ実行 (初回はWASMの最適化などが走る場合があるため)
  adapter.tune(testVector);

  // パフォーマンス計測
  const ITERATIONS = 100000;
  console.log(`\n⏳ ${ITERATIONS.toLocaleString()} 回の推論を実行中...`);
  
  const start = performance.now();
  let finalResult: Float32Array | null = null;
  
  for (let i = 0; i < ITERATIONS; i++) {
    finalResult = adapter.tune(testVector);
  }
  
  const end = performance.now();
  const timeMs = end - start;

  console.log(`📤 最終出力ベクトル (${outputDim}次元):`, finalResult);
  console.log(`\n⏱️ 合計時間: ${timeMs.toFixed(2)} ms`);
  console.log(`⚡ 1推論あたりの時間: ${(timeMs / ITERATIONS * 1000).toFixed(4)} マイクロ秒`);
  console.log(`🔥 推論スループット: ${Math.floor(ITERATIONS / (timeMs / 1000)).toLocaleString()} 回 / 秒`);
  
  console.log("\n💡 考察:");
  console.log("TensorFlow.jsなどの重いライブラリをロードせずとも、数十〜数百次元のMLPであればWASMとTypedArrayの組み合わせで極めて高速（毎秒数百万回レベル）に推論可能です！");
}

runMlpDemo().catch(console.error);
