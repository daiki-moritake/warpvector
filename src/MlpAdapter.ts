import { initWasm, ensureWasmMemory, getWasmMemory } from "./wasm/wasm-loader";
import { Activation } from "./utils";

/**
 * MLPの1層を定義するインターフェース
 */
export interface MlpLayer {
  /** 変換行列 W */
  matrix: number[][] | Float32Array;
  /** バイアス b */
  bias: number[] | Float32Array;
  /** 活性化関数 */
  activation: Activation;
}

function getActivationId(activation: Activation): number {
  switch (activation) {
    case "linear": return 0;
    case "relu": return 1;
    case "sigmoid": return 2;
    case "tanh": return 3;
    default: return 0;
  }
}

/**
 * MlpAdapter は WASM を使用して超高速に非線形な多層推論を行うラッパーです。
 */
export class MlpAdapter {
  private layers: MlpLayer[];
  private wasmInstance: WebAssembly.Instance | null = null;
  
  // WASMのポインタと設定値
  private isWasmReady = false;
  private inputDim = 0;
  private outputDim = 0;
  private numLayers = 0;
  
  private inputPtr = 0;
  private outputPtr = 0;
  private weightsPtr = 0;
  private layerDimsPtr = 0;
  private activationsPtr = 0;
  private bufferPtr = 0;

  constructor(layers: MlpLayer[]) {
    if (layers.length === 0) {
      throw new Error("MlpAdapter requires at least one layer.");
    }
    this.layers = layers;
    this.numLayers = layers.length;
  }

  /**
   * WASMの初期化と、MLP構造をWASMメモリに書き込む準備を行います。
   * インスタンス作成後に必ず呼び出してください。
   */
  public async init(): Promise<void> {
    this.wasmInstance = await initWasm();
    if (!this.wasmInstance) {
      throw new Error("Failed to initialize WASM for MlpAdapter.");
    }

    const memory = getWasmMemory()!;

    // 次元数の検証と計算
    let sDim = 0;
    let tDim = 0;
    let maxDim = 0;
    let totalWeights = 0;

    const layerDims: number[] = [];
    const activations: number[] = [];

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      let rows, cols;
      if (layer.matrix instanceof Float32Array) {
        rows = layer.bias.length;
        cols = layer.matrix.length / rows;
      } else {
        rows = layer.matrix.length;
        cols = layer.matrix[0].length;
      }

      if (i === 0) {
        sDim = cols;
        layerDims.push(sDim);
      } else if (cols !== tDim) {
        throw new Error(`Dimension mismatch at layer ${i}: expected input dim ${tDim}, got ${cols}`);
      }
      
      tDim = rows;
      layerDims.push(tDim);
      
      if (sDim > maxDim) maxDim = sDim;
      if (tDim > maxDim) maxDim = tDim;

      totalWeights += rows * cols + rows; // 行列要素 + バイアス要素
      activations.push(getActivationId(layer.activation));
    }

    this.inputDim = layerDims[0];
    this.outputDim = layerDims[layerDims.length - 1];

    // メモリレイアウトの計算 (各ポインタのオフセット)
    // 1. input (inputDim * 4)
    // 2. output (outputDim * 4)
    // 3. layerDims ((numLayers + 1) * 4)
    // 4. activations (numLayers * 4)
    // 5. weights (totalWeights * 4)
    // 6. buffer (最大層の次元 * 4 バイト を2つのバッファ用に x2 -> maxDim * 8。念の為4096足す)
    
    // 1ページ分(65536バイト)はWASM側の静的データやアロケータの領域として避け、
    // 安全な場所からレイアウトを開始する。
    let offset = 65536;
    this.inputPtr = offset; offset += this.inputDim * 4;
    this.outputPtr = offset; offset += this.outputDim * 4;
    this.layerDimsPtr = offset; offset += (this.numLayers + 1) * 4;
    this.activationsPtr = offset; offset += this.numLayers * 4;
    
    // アラインメント(Float32用)
    if (offset % 4 !== 0) offset += 4 - (offset % 4);
    this.weightsPtr = offset; offset += totalWeights * 4;
    this.bufferPtr = offset; offset += maxDim * 8 + 8192; // +8192はWASM側でのバッファBのオフセット(4096)用

    const requiredBytes = offset;
    ensureWasmMemory(requiredBytes);

    // データの書き込み
    const f32 = new Float32Array(memory.buffer);
    const i32 = new Int32Array(memory.buffer);

    // layerDims
    for (let i = 0; i < layerDims.length; i++) {
      i32[(this.layerDimsPtr / 4) + i] = layerDims[i];
    }
    // activations
    for (let i = 0; i < activations.length; i++) {
      i32[(this.activationsPtr / 4) + i] = activations[i];
    }
    // weights & bias
    let wIdx = this.weightsPtr / 4;
    for (let i = 0; i < this.numLayers; i++) {
      const layer = this.layers[i];
      let rows, cols;
      if (layer.matrix instanceof Float32Array) {
        rows = layer.bias.length;
        cols = layer.matrix.length / rows;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            f32[wIdx++] = layer.matrix[r * cols + c];
          }
          f32[wIdx++] = layer.bias[r];
        }
      } else {
        rows = layer.matrix.length;
        cols = layer.matrix[0].length;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            f32[wIdx++] = layer.matrix[r][c];
          }
          f32[wIdx++] = layer.bias[r];
        }
      }
    }

    this.isWasmReady = true;
  }

  /**
   * 入力ベクトルに対してMLPの推論を実行します。
   * @param vector 入力ベクトル
   * @returns 出力ベクトル (Float32Array)
   */
  public tune(vector: number[] | Float32Array): Float32Array {
    if (!this.isWasmReady || !this.wasmInstance) {
      throw new Error("MlpAdapter is not initialized. Call await init() first.");
    }
    if (vector.length !== this.inputDim) {
      throw new Error(`Input dimension mismatch. Expected ${this.inputDim}, got ${vector.length}`);
    }

    const memory = getWasmMemory()!;
    const f32 = new Float32Array(memory.buffer);

    // 入力ベクトルの書き込み
    const inIdx = this.inputPtr / 4;
    for (let i = 0; i < vector.length; i++) {
      f32[inIdx + i] = vector[i];
    }

    // WASMの呼び出し
    const mlpInferenceWasm = this.wasmInstance.exports.mlpInferenceWasm as CallableFunction;
    mlpInferenceWasm(
      this.inputPtr,
      this.outputPtr,
      this.weightsPtr,
      this.layerDimsPtr,
      this.activationsPtr,
      this.numLayers,
      this.bufferPtr
    );

    // 結果の読み取り
    const result = new Float32Array(this.outputDim);
    const outIdx = this.outputPtr / 4;
    for (let i = 0; i < this.outputDim; i++) {
      result[i] = f32[outIdx + i];
    }

    return result;
  }
}
