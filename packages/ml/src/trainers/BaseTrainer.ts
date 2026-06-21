import {
  initWasm,
  wasmMutex,
  assertDimension,
  applyAffine,
} from "@warpvector/core";

import { AbstractAdamTrainer } from "./AbstractAdamTrainer";
import { BaseTrainingOptions } from "./types";

export type { BaseTrainingOptions };
export { AbstractAdamTrainer };


/**
 * Adam オプティマイザを用いた学習のための共通基底クラス。
 *
 * @template TExample 学習に用いるデータペアの型 (入力と理想の出力のペアなど)
 * @template TResult 最終的に学習結果として出力される重み (行列やバイアス) の型
 */
export abstract class BaseTrainer<
  TExample,
  TResult,
> extends AbstractAdamTrainer<TResult> {
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
   * 学習用のサンプルデータを追加します。
   * 次元数がソース/ターゲットと一致しない場合はエラーとなります。
   *
   * @param {TExample} example 追加するサンプルデータ
   * @throws {Error} 次元数が一致しない場合にスローされます。
   */
  public addExample(example: TExample): void {
    const { source, target } = this.getInputs(example);
    assertDimension(
      source,
      this.sourceDimension,
      "BaseTrainer.addExample source",
    );
    assertDimension(
      target,
      this.targetDimension,
      "BaseTrainer.addExample target",
    );
    this.examples.push(example);
  }

  /**
   * 追加されたサンプルデータを用いて学習を実行します。
   * 指定されたエポック数だけ Adam によるパラメータ更新を行います。
   * パフォーマンスのため、内部で WebAssembly (WASM) を使用します。
   *
   * @param {BaseTrainingOptions} [options={}] 学習のハイパーパラメータオプション
   * @returns {Promise<TResult>} 学習済みの重みを返します。
   * @throws {Error} サンプルデータが追加されていない場合にスローされます。
   */
  public async train(options: BaseTrainingOptions = {}): Promise<TResult> {
    return wasmMutex.runExclusive(async () => {
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
            this.t,
          );
        }
      }

      return this.toWeights(flatMatrix, bias);
    });
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
            t,
          );
        }
      }

      let currentLoss = 0;
      for (const example of this.examples) {
        const { source, target } = this.getInputs(example);
        const pred = new Float32Array(tDim);
        applyAffine(flatMatrix, bias, source, pred, sDim, tDim);
        for (let i = 0; i < tDim; i++) {
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
   * バックワード処理は WASM バックエンドへオフロードされます。
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
    t: number,
  ): void {
    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;

    const pred = new Float32Array(tDim);
    applyAffine(matrix, bias, x, pred, sDim, tDim);

    const outputGradients = new Float32Array(tDim);
    for (let i = 0; i < tDim; i++) {
      outputGradients[i] = pred[i] - y[i];
    }

    this.applyAdamToAffine(
      matrix,
      bias,
      mMatrix,
      vMatrix,
      mBias,
      vBias,
      x,
      outputGradients,
      lr,
      reg,
      t,
    );
  }
}
