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
  safeJsonParse,
  assertObject,
  assertArray,
  assertNumberArray,
  assertType,
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

  // init() 時に永続化されるWASMメモリのポインタ
  private persistentWeightsPtr = 0;
  private persistentLayerDimsPtr = 0;
  private persistentActivationsPtr = 0;

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
   *
   * 重み・layerDims・activations はこのメソッド内で WASM メモリの永続領域に
   * 一度だけ書き込まれます。tune() の呼び出しでは入出力バッファのみが
   * スタック管理され、重みの再転送は行われません。
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

    // --- 重みを WASM メモリの永続領域に書き込む ---
    this.persistentWeightsPtr = allocateWasmMemory(this.totalWeights * 4);
    this.persistentLayerDimsPtr = allocateWasmMemory((this.numLayers + 1) * 4);
    this.persistentActivationsPtr = allocateWasmMemory(this.numLayers * 4);

    this.writeWeightsToWasm();

    this.isWasmReady = true;
  }

  /**
   * 現在の layers 配列の重み・layerDims・activations を永続化済みの
   * WASM メモリ領域に書き込みます。init() および setLayerWeights() で使用します。
   */
  private writeWeightsToWasm(): void {
    const memory = this.wasmInstance!.exports.memory as WebAssembly.Memory;
    const memoryBuffer = memory.buffer;
    const f32 = new Float32Array(memoryBuffer);
    const i32 = new Int32Array(memoryBuffer);

    // layerDims
    for (let i = 0; i < this.layerDims.length; i++) {
      i32[this.persistentLayerDimsPtr / 4 + i] = this.layerDims[i];
    }
    // activations
    for (let i = 0; i < this.activations.length; i++) {
      i32[this.persistentActivationsPtr / 4 + i] = this.activations[i];
    }
    // weights & bias (interleaved: row weights then row bias for each row)
    let wIdx = this.persistentWeightsPtr / 4;
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
  }

  /**
   * 特定のレイヤーの重みを実行時に更新します。
   * WASM メモリの永続領域も同期的に再書き込みされます。
   *
   * @param layerIndex 更新するレイヤーのインデックス
   * @param layer 新しいレイヤーデータ
   */
  public setLayerWeights(layerIndex: number, layer: MlpLayer): void {
    if (layerIndex < 0 || layerIndex >= this.numLayers) {
      throw new Error(`Layer index ${layerIndex} out of range (0-${this.numLayers - 1})`);
    }
    this.layers[layerIndex] = layer;

    // WASM が初期化済みなら永続領域を再書き込み
    if (this.isWasmReady) {
      this.writeWeightsToWasm();
    }
  }

  /**
   * ニューラルネットワークの順伝播を実行し、結果を返します。
   * (WarpAdapter の実装として、predict の代わりに tune を提供します)
   *
   * 重み・layerDims・activations は init() 時に永続化された WASM メモリを
   * 直接参照します。入出力バッファと中間バッファのみがスタック管理されます。
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
      // 入出力と中間バッファのみをスタック上に確保
      const inputPtr = allocateWasmMemory(this.inputDim * 4);
      const outputPtr = allocateWasmMemory(this.outputDim * 4);
      const bufferPtr = allocateWasmMemory(this.maxDim * 4);
      const bufBPtr = allocateWasmMemory(this.maxDim * 4);

      // 入力ベクトルの書き込み
      writeFloat32ArrayToWasm(memory, input, inputPtr);

      // WASMの呼び出し（重みは永続化済みのポインタを使用）
      const mlpInferenceWasm = this.wasmInstance!.exports
        .mlpInferenceWasm as CallableFunction;
      mlpInferenceWasm(
        inputPtr,
        outputPtr,
        this.persistentWeightsPtr,
        this.persistentLayerDimsPtr,
        this.persistentActivationsPtr,
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
    const data = assertObject(
      safeJsonParse(stateJson, "MlpAdapter"),
      "root",
    );
    const rawLayers = assertArray(data.layers, "layers");
    const layers: MlpLayer[] = rawLayers.map((rawLayer: unknown, i: number) => {
      const l = assertObject(rawLayer, `layers[${i}]`);
      assertType(l.activation, "string", `layers[${i}].activation`);
      const bias = assertNumberArray(l.bias, `layers[${i}].bias`);

      // matrix は 2D配列 or flat 1D配列の両方をサポート
      let matrix: number[][] | Float32Array;
      if (Array.isArray(l.matrix) && Array.isArray(l.matrix[0])) {
        // 2D 配列
        matrix = (l.matrix as unknown[][]).map(
          (row: unknown, j: number) =>
            assertNumberArray(row, `layers[${i}].matrix[${j}]`),
        );
      } else {
        matrix = new Float32Array(
          assertNumberArray(l.matrix, `layers[${i}].matrix`),
        );
      }

      return {
        matrix,
        bias: new Float32Array(bias),
        activation: l.activation as Activation,
      };
    });
    return new MlpAdapter(layers);
  }
}
