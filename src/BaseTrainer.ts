import {
  initWasm,
  getWasmInstance,
  getWasmMemory,
  ensureWasmMemory,
} from "./wasm/wasm-loader";
import { assertDimension } from "./utils";

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
 * Adam最適化のステート変数を管理する共通基底クラス
 */
export abstract class AbstractAdamTrainer {
  protected t: number = 0;
  protected mW!: Float32Array;
  protected vW!: Float32Array;
  protected mb!: Float32Array;
  protected vb!: Float32Array;

  protected initAdamState(sDim: number, tDim: number): void {
    if (!this.mW || this.mW.length !== sDim * tDim) {
      this.mW = new Float32Array(sDim * tDim);
      this.vW = new Float32Array(sDim * tDim);
      this.mb = new Float32Array(tDim);
      this.vb = new Float32Array(tDim);
      this.t = 0;
    }
    assertDimension(this.mW, sDim * tDim, "AdamState mW");
  }
}

/**
 * 確率的勾配降下法 (SGD + Momentum) を用いた学習のための共通基底クラス。
 *
 * @template TExample 学習に用いるデータペアの型 (入力と理想の出力のペアなど)
 * @template TResult 最終的に学習結果として出力される重み (行列やバイアス) の型
 */
export abstract class BaseTrainer<TExample, TResult> extends AbstractAdamTrainer {
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
    assertDimension(source, this.sourceDimension, "BaseTrainer.addExample source");
    assertDimension(target, this.targetDimension, "BaseTrainer.addExample target");
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

    this.initAdamState(sDim, tDim);

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const example of this.examples) {
        this.t++;
        const { source, target } = this.getInputs(example);
        this.adamStep(
          flatMatrix,
          bias,
          this.mW,
          this.vW,
          this.mb,
          this.vb,
          source,
          target,
          lr,
          reg,
          this.t
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
      const mMatrix = new Float32Array(tDim * sDim);
      const vMatrix = new Float32Array(tDim * sDim);
      const mBias = new Float32Array(tDim);
      const vBias = new Float32Array(tDim);
      let t = 0;

      for (let epoch = 0; epoch < testEpochs; epoch++) {
        for (const example of this.examples) {
          t++;
          const { source, target } = this.getInputs(example);
          this.adamStep(
            flatMatrix,
            bias,
            mMatrix,
            vMatrix,
            mBias,
            vBias,
            source,
            target,
            lr,
            reg,
            t
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
   * Adam オプティマイザによる1ステップのパラメータ更新を実行します。
   * In-place (破壊的) に `matrix` と `bias` を更新します。
   * WASM 版の Adam 実装ができるまではネイティブ JS で処理します。
   */
  protected adamStep(
    matrix: Float32Array,
    bias: Float32Array,
    mMatrix: Float32Array,
    vMatrix: Float32Array,
    mBias: Float32Array,
    vBias: Float32Array,
    x: number[] | Float32Array,
    y: number[] | Float32Array,
    lr: number,
    reg: number,
    t: number
  ): void {
    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

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
      mBias[i] = beta1 * mBias[i] + (1 - beta1) * bGrad;
      vBias[i] = beta2 * vBias[i] + (1 - beta2) * (bGrad * bGrad);
      const mHatB = mBias[i] / (1 - Math.pow(beta1, t));
      const vHatB = vBias[i] / (1 - Math.pow(beta2, t));
      
      bias[i] -= lr * mHatB / (Math.sqrt(vHatB) + epsilon);

      const rowOffset = i * sDim;
      for (let j = 0; j < sDim; j++) {
        const wIdx = rowOffset + j;
        const wGrad = error * x[j] + reg * matrix[wIdx];
        
        mMatrix[wIdx] = beta1 * mMatrix[wIdx] + (1 - beta1) * wGrad;
        vMatrix[wIdx] = beta2 * vMatrix[wIdx] + (1 - beta2) * (wGrad * wGrad);
        const mHatW = mMatrix[wIdx] / (1 - Math.pow(beta1, t));
        const vHatW = vMatrix[wIdx] / (1 - Math.pow(beta2, t));

        matrix[wIdx] -= lr * mHatW / (Math.sqrt(vHatW) + epsilon);
      }
    }
  }
}
