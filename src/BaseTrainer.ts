import {
  initWasm,
  getWasmInstance,
  getWasmMemory,
  ensureWasmMemory,
} from "./wasm/wasm-loader";

/**
 * 基本的な学習オプションを定義するインターフェース。
 * 勾配降下法における各種ハイパーパラメータを設定します。
 */
export interface BaseTrainingOptions {
  /** 学習率 (Learning Rate)。1ステップで重みをどれだけ更新するか。デフォルト: 0.01 */
  learningRate?: number;
  /** 学習のエポック数 (データセット全体を何回繰り返して学習するか)。デフォルト: 100 */
  epochs?: number;
  /** L2正則化の強さ。過学習を防ぐために使用します。デフォルト: 0.001 */
  regularization?: number;
  /** モーメンタム (Momentum)。前回の更新量をどれだけ引き継ぐか。デフォルト: 0.9 */
  momentum?: number;
  /** trueの場合、事前に数エポックのテストランを行い、最適な学習率を自動探索します。デフォルト: false */
  autoTune?: boolean;
}

/**
 * 確率的勾配降下法 (SGD + Momentum) を用いた学習のための共通基底クラス。
 *
 * @template TExample 学習に用いるデータペアの型 (入力と理想の出力のペアなど)
 * @template TResult 最終的に学習結果として出力される重み (行列やバイアス) の型
 */
export abstract class BaseTrainer<TExample, TResult> {
  /** 学習用サンプルの配列 */
  protected examples: TExample[] = [];

  /** 入力ベクトル（ソース）の次元数 */
  protected abstract get sourceDimension(): number;
  /** 出力ベクトル（ターゲット）の次元数 */
  protected abstract get targetDimension(): number;

  /**
   * サンプルデータから、学習アルゴリズムに渡すソースベクトルとターゲットベクトルを抽出します。
   * @param {TExample} example 学習サンプルのデータ
   * @returns {{ source: number[] | Float32Array; target: number[] | Float32Array }}
   */
  protected abstract getInputs(example: TExample): {
    source: number[] | Float32Array;
    target: number[] | Float32Array;
  };
  /**
   * 学習済みの1次元行列とバイアスを、最終的な結果の型 (TResult) に変換します。
   * @param {Float32Array} flatMatrix 学習済みのフラット化された行列
   * @param {Float32Array} bias 学習済みのバイアスベクトル
   * @returns {TResult} 変換された重みデータ
   */
  protected abstract toWeights(
    flatMatrix: Float32Array,
    bias: Float32Array,
  ): TResult;

  /**
   * 学習用のサンプルデータを追加します。
   * 次元数がソース/ターゲットと一致しない場合はエラーとなります。
   *
   * @param {TExample} example 追加するサンプルデータ
   * @throws {Error} 次元数が一致しない場合にスローされます。
   */
  public addExample(example: TExample): void {
    const { source, target } = this.getInputs(example);
    if (source.length !== this.sourceDimension) {
      throw new Error(
        `Source dimension mismatch. Expected ${this.sourceDimension}.`,
      );
    }
    if (target.length !== this.targetDimension) {
      throw new Error(
        `Target dimension mismatch. Expected ${this.targetDimension}.`,
      );
    }
    this.examples.push(example);
  }

  /**
   * 追加されたサンプルデータを用いて学習を実行します。
   * 指定されたエポック数だけ SGD + Momentum によるパラメータ更新を行います。
   * パフォーマンスのため、可能であれば内部で WebAssembly (WASM) を使用します。
   *
   * @param {BaseTrainingOptions} [options={}] 学習のハイパーパラメータオプション
   * @returns {Promise<TResult>} 学習済みの重みを返します。
   * @throws {Error} サンプルデータが追加されていない場合にスローされます。
   */
  public async train(options: BaseTrainingOptions = {}): Promise<TResult> {
    await initWasm();

    if (this.examples.length === 0) {
      throw new Error("No training examples provided.");
    }

    if (options.autoTune) {
      options.learningRate = this.findBestLearningRate(options);
      options.autoTune = false;
    }

    const lr = options.learningRate ?? 0.01;
    const epochs = options.epochs ?? 100;
    const reg = options.regularization ?? 0.001;
    const momentum = options.momentum ?? 0.9;

    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;

    const flatMatrix = new Float32Array(tDim * sDim);
    for (let i = 0; i < tDim; i++) {
      if (i < sDim) {
        flatMatrix[i * sDim + i] = 1.0;
      }
    }
    const bias = new Float32Array(tDim);

    const vMatrix = new Float32Array(tDim * sDim);
    const vBias = new Float32Array(tDim);

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const example of this.examples) {
        const { source, target } = this.getInputs(example);
        this.sgdMomentumStep(
          flatMatrix,
          bias,
          vMatrix,
          vBias,
          source,
          target,
          lr,
          reg,
          momentum,
        );
      }
    }

    return this.toWeights(flatMatrix, bias);
  }

  /**
   * サンプルデータに対する短時間のテストランを行い、最も損失(Loss)が小さくなる最適な学習率を自動探索します。
   * `options.autoTune` が true の場合に `train` メソッド内で自動的に呼び出されます。
   *
   * @param {BaseTrainingOptions} options 現在の学習オプション
   * @returns {number} 探索された最適な学習率
   */
  protected findBestLearningRate(options: BaseTrainingOptions): number {
    const candidateLrs = [0.1, 0.05, 0.01, 0.005, 0.001];
    let bestLr = options.learningRate ?? 0.01;
    let minLoss = Infinity;

    const testEpochs = Math.min(10, options.epochs ?? 100);
    const reg = options.regularization ?? 0.001;
    const momentum = options.momentum ?? 0.9;

    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;

    for (const lr of candidateLrs) {
      const flatMatrix = new Float32Array(tDim * sDim);
      for (let i = 0; i < tDim; i++) {
        if (i < sDim) flatMatrix[i * sDim + i] = 1.0;
      }
      const bias = new Float32Array(tDim);
      const vMatrix = new Float32Array(tDim * sDim);
      const vBias = new Float32Array(tDim);

      for (let epoch = 0; epoch < testEpochs; epoch++) {
        for (const example of this.examples) {
          const { source, target } = this.getInputs(example);
          this.sgdMomentumStep(
            flatMatrix,
            bias,
            vMatrix,
            vBias,
            source,
            target,
            lr,
            reg,
            momentum,
          );
        }
      }

      let currentLoss = 0;
      for (const example of this.examples) {
        const { source, target } = this.getInputs(example);
        const pred = new Float32Array(tDim);
        for (let i = 0; i < tDim; i++) {
          let sum = 0;
          for (let j = 0; j < sDim; j++) {
            sum += flatMatrix[i * sDim + j] * source[j];
          }
          pred[i] = sum + bias[i];
          const diff = pred[i] - target[i];
          currentLoss += diff * diff;
        }
      }

      if (currentLoss < minLoss) {
        minLoss = currentLoss;
        bestLr = lr;
      }
    }

    return bestLr;
  }

  /**
   * SGD + Momentum アルゴリズムによる1ステップのパラメータ更新を実行します。
   * In-place (破壊的) に `matrix` と `bias` を更新します。
   *
   * @param {Float32Array} matrix 現在の変換行列 (1次元フラット配列)
   * @param {Float32Array} bias 現在のバイアスベクトル
   * @param {Float32Array} vMatrix 行列のモメンタム (速度)
   * @param {Float32Array} vBias バイアスのモメンタム (速度)
   * @param {number[] | Float32Array} x 入力ベクトル (ソース)
   * @param {number[] | Float32Array} y 理想の出力ベクトル (ターゲット)
   * @param {number} lr 学習率
   * @param {number} reg L2正則化係数
   * @param {number} momentum モメンタム係数
   */
  protected sgdMomentumStep(
    matrix: Float32Array,
    bias: Float32Array,
    vMatrix: Float32Array,
    vBias: Float32Array,
    x: number[] | Float32Array,
    y: number[] | Float32Array,
    lr: number,
    reg: number,
    momentum: number,
  ): void {
    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;
    const instance = getWasmInstance();

    const requiredBytes =
      sDim * tDim * 4 +
      tDim * 4 +
      sDim * tDim * 4 +
      tDim * 4 +
      sDim * 4 +
      tDim * 4 +
      tDim * 4;

    if (instance && ensureWasmMemory(requiredBytes)) {
      const wasmMemory = getWasmMemory()!;
      const f32Mem = new Float32Array(wasmMemory.buffer);

      let offset = 0;
      const matrixOffset = offset;
      offset += sDim * tDim;
      const biasOffset = offset;
      offset += tDim;
      const vMatrixOffset = offset;
      offset += sDim * tDim;
      const vBiasOffset = offset;
      offset += tDim;
      const xOffset = offset;
      offset += sDim;
      const yOffset = offset;
      offset += tDim;
      const predOffset = offset;
      offset += tDim;

      f32Mem.set(matrix, matrixOffset);
      f32Mem.set(bias, biasOffset);
      f32Mem.set(vMatrix, vMatrixOffset);
      f32Mem.set(vBias, vBiasOffset);
      f32Mem.set(x as Float32Array, xOffset);
      f32Mem.set(y as Float32Array, yOffset);

      const sgdMomentumStepWasm = instance.exports
        .sgdMomentumStepWasm as CallableFunction;

      sgdMomentumStepWasm(
        matrixOffset * 4,
        biasOffset * 4,
        vMatrixOffset * 4,
        vBiasOffset * 4,
        xOffset * 4,
        yOffset * 4,
        lr,
        reg,
        momentum,
        sDim,
        tDim,
        predOffset * 4,
      );

      matrix.set(f32Mem.subarray(matrixOffset, matrixOffset + sDim * tDim));
      bias.set(f32Mem.subarray(biasOffset, biasOffset + tDim));
      vMatrix.set(f32Mem.subarray(vMatrixOffset, vMatrixOffset + sDim * tDim));
      vBias.set(f32Mem.subarray(vBiasOffset, vBiasOffset + tDim));

      return;
    }

    // WASMが使えない場合のJSフォールバック
    const pred = new Float32Array(tDim);

    for (let i = 0; i < tDim; i++) {
      let sum = 0;
      const rowOffset = i * sDim;
      for (let j = 0; j < sDim; j++) {
        sum += matrix[rowOffset + j] * x[j];
      }
      pred[i] = sum + bias[i];
    }

    for (let i = 0; i < tDim; i++) {
      const error = pred[i] - y[i];

      const bGrad = error;
      vBias[i] = momentum * vBias[i] - lr * bGrad;
      bias[i] += vBias[i];

      const rowOffset = i * sDim;
      for (let j = 0; j < sDim; j++) {
        const wIdx = rowOffset + j;
        const wGrad = error * x[j] + reg * matrix[wIdx];
        vMatrix[wIdx] = momentum * vMatrix[wIdx] - lr * wGrad;
        matrix[wIdx] += vMatrix[wIdx];
      }
    }
  }
}
