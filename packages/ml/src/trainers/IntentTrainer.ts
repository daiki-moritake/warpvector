import { IntentWeights } from "@warpvector/core";
import { BaseTrainer, BaseTrainingOptions } from "../trainers/BaseTrainer";
import { initWasm } from "@warpvector/core";
import { assertDimension, getFlatMatrixAndBias, applyAffine } from "@warpvector/core";

/**
 * 学習データのペア（ベースのクエリベクトルと、目標となる結果ベクトル）
 * @interface TrainingExample
 */
export interface TrainingExample {
  /** 入力ベクトル（元の検索クエリなどの埋め込み表現） */
  input: number[] | Float32Array;
  /** 目標ベクトル（ユーザーが理想とする結果の埋め込み表現） */
  target: number[] | Float32Array;
}

/**
 * 学習時の最適化オプション
export interface TrainingOptions extends BaseTrainingOptions {}

/**
 * IntentTrainer のオンライン学習オプション
 */
export interface IntentOnlineOptions {
  /** 1ステップの学習率 (デフォルト: 0.01) */
  learningRate?: number;
  /** L2正則化の強さ (デフォルト: 0.001) */
  regularization?: number;
}

/**
 * サンプルデータから最適な `IntentWeights` (行列Wとバイアスb) を
 * 確率的勾配降下法 (SGD + Momentum) により自動学習するトレーナークラス。
 * 内部で Float32Array のフラット化を行い、極限のパフォーマンスを引き出します。
 *
 * @example
 * const trainer = new IntentTrainer(1536);
 * trainer.addExample({ input: [...], target: [...] });
 * const weights = await trainer.train({ autoTune: true });
 */
export class IntentTrainer extends BaseTrainer<TrainingExample, IntentWeights> {
  private dimension: number;

  /**
   * IntentTrainer のインスタンスを作成します。
   * @param {number} dimension ベクトルの次元数（入力・出力ともに同じ次元数となります）
   */
  constructor(dimension: number) {
    super();
    this.dimension = dimension;
    this.initAdamState(dimension, dimension);
  }

  protected get sourceDimension(): number {
    return this.dimension;
  }

  protected get targetDimension(): number {
    return this.dimension;
  }

  protected getInputs(example: TrainingExample): {
    source: number[] | Float32Array;
    target: number[] | Float32Array;
  } {
    return { source: example.input, target: example.target };
  }



  /**
   * オンライン学習 (フィードバックループ) 用のメソッド。
   * ユーザーのクリックなどの 1 回のフィードバックからリアルタイムに重みを微調整します。
   *
   * @param {IntentWeights} currentWeights - 現在の重み
   * @param {TrainingExample} example - アンカー、正解を含む学習データ
   * @param {IntentOnlineOptions} [options={}] - 学習オプション
   * @returns {IntentWeights} 微調整された新しい重み
   */
  public async updateOnline(
    currentWeights: IntentWeights,
    example: TrainingExample,
    options: IntentOnlineOptions = {},
  ): Promise<IntentWeights> {
    const learningRate = options.learningRate ?? 0.01;
    const regularization = options.regularization ?? 0.001;
    await initWasm();

    assertDimension(
      example.input,
      this.dimension,
      "IntentTrainer.addExample input",
    );
    assertDimension(
      example.target,
      this.dimension,
      "IntentTrainer.addExample target",
    );

    const dim = this.dimension;
    const { flatMatrix, bias } = getFlatMatrixAndBias(
      currentWeights,
      dim,
      "updateOnline Matrix",
    );

    // 1. Forward pass (アフィン変換のみ。活性化関数は適用前)
    const warpedInput = new Float32Array(dim);
    applyAffine(flatMatrix, bias, example.input, warpedInput, dim);

    // 誤差の計算 (dL/dY = Y - T)
    const outputGradients = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      outputGradients[i] = warpedInput[i] - example.target[i];
    }

    // オンライン更新では内部の Adam ステートを使用する
    this.t++;
    this.applyAdamToAffine(
      flatMatrix,
      bias,
      this.mW,
      this.vW,
      this.mb,
      this.vb,
      example.input,
      outputGradients,
      learningRate,
      regularization,
      this.t,
    );

    const newWeights = this.toWeights(flatMatrix, bias);
    if (currentWeights.routingVector) {
      newWeights.routingVector = [...currentWeights.routingVector];
    }
    return newWeights;
  }
}
