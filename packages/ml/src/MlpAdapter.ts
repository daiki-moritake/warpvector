import {
  type WarpAdapter,
  type Activation,
  assertDimension,
  initWasm,
  ensureWasmMemory,
  writeFloat32ArrayToWasm,
  allocateWasmMemory,
  withWasmMemoryStack,
  readFloat32ArrayFromWasm,
} from "@warpvector/core";

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
    case "linear":
      return 0;
    case "relu":
      return 1;
    case "sigmoid":
      return 2;
    case "tanh":
      return 3;
    default:
      return 0;
  }
}

/**
 * MlpAdapter は WASM を使用して超高速に非線形な多層推論を行うラッパーです。
 *
 * @example
 * const mlp = new MlpAdapter([{ inputDim: 1536, outputDim: 128, activation: "relu" }]);
 * const output = mlp.tune(inputVector);
 */
export class MlpAdapter implements WarpAdapter {
  private layers: MlpLayer[];
  private wasmInstance: WebAssembly.Instance | null = null;

  // WASMのポインタと設定値
  private isWasmReady = false;
  private inputDim = 0;
  private outputDim = 0;
  private numLayers = 0;

  // 計算に必要な中間パラメータのキャッシュ
  private layerDims: number[] = [];
  private activations: number[] = [];
  private totalWeights = 0;
  private maxDim = 0;

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

    // 次元数の検証と計算
    let sDim = 0;
    let tDim = 0;
    this.maxDim = 0;
    this.totalWeights = 0;

    this.layerDims = [];
    this.activations = [];

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
        this.layerDims.push(sDim);
      } else if (cols !== tDim) {
        throw new Error(
          `Dimension mismatch at layer ${i}: expected input dim ${tDim}, got ${cols}`,
        );
      }

      tDim = rows;
      this.layerDims.push(tDim);

      if (sDim > this.maxDim) this.maxDim = sDim;
      if (tDim > this.maxDim) this.maxDim = tDim;

      this.totalWeights += rows * cols + rows; // 行列要素 + バイアス要素
      this.activations.push(getActivationId(layer.activation));
    }

    this.inputDim = this.layerDims[0];
    this.outputDim = this.layerDims[this.layerDims.length - 1];

    this.isWasmReady = true;
  }

  /**
   * ニューラルネットワークの順伝播を実行し、結果を返します。
   * (WarpAdapter の実装として、predict の代わりに tune を提供します)
   *
   * @param input 入力ベクトル
   * @returns 推論結果ベクトル
   */
  public tune(input: number[] | Float32Array): Float32Array {
    if (!this.isWasmReady || !this.wasmInstance) {
      throw new Error(
        "MlpAdapter is not initialized. Call await init() first.",
      );
    }
    assertDimension(input, this.inputDim, "MlpAdapter.tune");

    const memory = this.wasmInstance.exports.memory as WebAssembly.Memory;
    return withWasmMemoryStack(() => {
      const inputPtr = allocateWasmMemory(this.inputDim * 4);
      const outputPtr = allocateWasmMemory(this.outputDim * 4);
      const layerDimsPtr = allocateWasmMemory((this.numLayers + 1) * 4);
      const activationsPtr = allocateWasmMemory(this.numLayers * 4);
      const weightsPtr = allocateWasmMemory(this.totalWeights * 4);
      const bufferPtr = allocateWasmMemory(this.maxDim * 4);
      const bufBPtr = allocateWasmMemory(this.maxDim * 4);

      // データの書き込み
      const memoryBuffer = memory.buffer;
      const f32 = new Float32Array(memoryBuffer);
      const i32 = new Int32Array(memoryBuffer);

      // layerDims
      for (let i = 0; i < this.layerDims.length; i++) {
        i32[layerDimsPtr / 4 + i] = this.layerDims[i];
      }
      // activations
      for (let i = 0; i < this.activations.length; i++) {
        i32[activationsPtr / 4 + i] = this.activations[i];
      }
      // weights & bias
      let wIdx = weightsPtr / 4;
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

      // 入力ベクトルの書き込み
      writeFloat32ArrayToWasm(memory, input, inputPtr);

      // WASMの呼び出し
      const mlpInferenceWasm = this.wasmInstance!.exports
        .mlpInferenceWasm as CallableFunction;
      mlpInferenceWasm(
        inputPtr,
        outputPtr,
        weightsPtr,
        layerDimsPtr,
        activationsPtr,
        this.numLayers,
        bufferPtr,
        bufBPtr,
      );

      // 結果の読み取り
      return readFloat32ArrayFromWasm(memory, outputPtr, this.outputDim);
    });
  }

  /**
   * 現在のMLP構造と重みをシリアライズして出力します。
   */
  public exportState(): string {
    return JSON.stringify({
      layers: this.layers.map((layer) => {
        let matrix: number[][] | number[];
        if (layer.matrix instanceof Float32Array) {
          matrix = Array.from(layer.matrix);
        } else {
          matrix = layer.matrix;
        }
        return {
          matrix,
          bias: Array.from(layer.bias),
          activation: layer.activation,
        };
      }),
    });
  }

  /**
   * シリアライズされた状態から MlpAdapter を復元します。
   * 注意: 復元後、再度 `await init()` を呼び出してWASMメモリを初期化する必要があります。
   */
  public static importState(stateJson: string): MlpAdapter {
    const data = JSON.parse(stateJson);
    const layers: MlpLayer[] = data.layers.map((l: any) => ({
      // Float32Array に戻すか、そのまま2D配列として扱う
      matrix: Array.isArray(l.matrix[0])
        ? l.matrix
        : new Float32Array(l.matrix),
      bias: new Float32Array(l.bias),
      activation: l.activation,
    }));
    return new MlpAdapter(layers);
  }
}
