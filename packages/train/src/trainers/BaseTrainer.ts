import { initWasm, wasmMutex } from "@warpvector/core";

import { AbstractAdamTrainer } from "./AbstractAdamTrainer";
import { BaseTrainingOptions } from "./types";

export type { BaseTrainingOptions };
export { AbstractAdamTrainer };

/**
 * Adam オプティマイザを用いた学習のための共通基底クラス。
 * 任意の損失関数 (MSE, Triplet, InfoNCE 等) をプラグインできる Template Method パターンを採用しています。
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
   * 学習用のサンプルデータを追加します。
   *
   * @param {TExample} example 追加するサンプルデータ
   */
  public addExample(example: TExample): void {
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

      const patience = options.patience;
      let bestLoss = Infinity;
      let patienceCounter = 0;

      for (let epoch = 0; epoch < epochs; epoch++) {
        for (const example of this.examples) {
          this.t++;
          this.adamStep(
            flatMatrix,
            bias,
            this.mW,
            this.vW,
            this.mb,
            this.vb,
            example,
            lr,
            reg,
            this.t,
            options, // allow passing custom options like temperature or margin
          );
        }

        if (patience !== undefined) {
          let currentLoss = 0;
          for (const example of this.examples) {
            currentLoss += this.calculateLoss(
              flatMatrix,
              bias,
              example,
              options,
            );
          }
          currentLoss /= this.examples.length || 1;

          if (currentLoss < bestLoss) {
            bestLoss = currentLoss;
            patienceCounter = 0;
          } else {
            patienceCounter++;
            if (patienceCounter >= patience) {
              options.onEarlyStopping?.(epoch + 1, patience);
              break;
            }
          }
        }
      }

      return this.toWeights(flatMatrix, bias);
    });
  }

  /**
   * メモリに全データを保持せず、ストリームやジェネレータからデータを逐次読み込んで学習を実行します。
   * 大規模なデータセット（数百万件など）で OOM を防ぐためのスケーラブルな学習メソッドです。
   *
   * @param dataFactory 1エポック分の学習データを生成する関数。AsyncIterable または Iterable を返します。
   * @param options 学習のハイパーパラメータオプション
   * @returns 学習済みの重みを返します。
   */
  public async trainFromGenerator(
    dataFactory: () => AsyncIterable<TExample> | Iterable<TExample>,
    options: BaseTrainingOptions = {},
  ): Promise<TResult> {
    return wasmMutex.runExclusive(async () => {
      await initWasm();

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

      const patience = options.patience;
      let bestLoss = Infinity;
      let patienceCounter = 0;

      for (let epoch = 0; epoch < epochs; epoch++) {
        let currentLossSum = 0;
        let exampleCount = 0;

        for await (const example of dataFactory()) {
          this.t++;
          this.adamStep(
            flatMatrix,
            bias,
            this.mW,
            this.vW,
            this.mb,
            this.vb,
            example,
            lr,
            reg,
            this.t,
            options,
          );

          if (patience !== undefined) {
            currentLossSum += this.calculateLoss(
              flatMatrix,
              bias,
              example,
              options,
            );
            exampleCount++;
          }
        }

        if (patience !== undefined && exampleCount > 0) {
          const avgLoss = currentLossSum / exampleCount;
          if (avgLoss < bestLoss) {
            bestLoss = avgLoss;
            patienceCounter = 0;
          } else {
            patienceCounter++;
            if (patienceCounter >= patience) {
              options.onEarlyStopping?.(epoch + 1, patience);
              break;
            }
          }
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
          this.adamStep(
            flatMatrix,
            bias,
            mMatrix,
            vMatrix,
            mBias,
            vBias,
            example,
            lr,
            reg,
            t,
            options,
          );
        }
      }

      let currentLoss = 0;
      for (const example of this.examples) {
        currentLoss += this.calculateLoss(flatMatrix, bias, example, options);
      }

      if (currentLoss < minLoss) {
        minLoss = currentLoss;
        bestLr = lr;
      }
    }

    return bestLr;
  }

  /**
   * 指定されたサンプルに対する損失 (Loss) を計算します。
   * サブクラスで独自の損失関数 (L2, マージンロス, クロスエントロピー等) を実装してください。
   *
   * @param {Float32Array} matrix 現在の重み行列
   * @param {Float32Array} bias 現在のバイアス
   * @param {TExample} example 評価するサンプル
   * @param {BaseTrainingOptions} [options] その他オプション
   * @returns {number} 計算された損失値
   */
  protected abstract calculateLoss(
    matrix: Float32Array,
    bias: Float32Array,
    example: TExample,
    options?: BaseTrainingOptions,
  ): number;

  /**
   * Adam オプティマイザによる1ステップのパラメータ更新を実行します。
   * サブクラスで、独自の勾配計算ロジックを実装し、内部で `applyAdamToAffine` を呼び出して
   * In-place (破壊的) に `matrix` と `bias` を更新してください。
   *
   * @param {Float32Array} matrix 更新対象の重み行列
   * @param {Float32Array} bias 更新対象のバイアス
   * @param {Float32Array} mMatrix Adam のモメンタムベクトル M (行列用)
   * @param {Float32Array} vMatrix Adam のモメンタムベクトル V (行列用)
   * @param {Float32Array} mBias Adam のモメンタムベクトル M (バイアス用)
   * @param {Float32Array} vBias Adam のモメンタムベクトル V (バイアス用)
   * @param {TExample} example 勾配計算の元となるサンプル
   * @param {number} lr 学習率
   * @param {number} reg L2正則化の強さ
   * @param {number} t 現在のタイムステップ
   * @param {BaseTrainingOptions} [options] その他オプション
   */
  protected abstract adamStep(
    matrix: Float32Array,
    bias: Float32Array,
    mMatrix: Float32Array,
    vMatrix: Float32Array,
    mBias: Float32Array,
    vBias: Float32Array,
    example: TExample,
    lr: number,
    reg: number,
    t: number,
    options?: BaseTrainingOptions,
  ): void;
}
