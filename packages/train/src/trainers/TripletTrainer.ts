import { IntentWeights, initWasm, wasmMutex, type TripletExample } from "@warpvector/core";
import {
  assertDimension,
  getFlatMatrixAndBias,
  applyAffine,
  innerProduct,
} from "@warpvector/core";
import { BaseTrainer } from "../trainers/BaseTrainer";


/**
 * TripletTrainer のオンライン学習オプション
 */
export interface TripletOnlineOptions {
  /** 1ステップの学習率 (デフォルト: 0.01) */
  learningRate?: number;
  /** トリプレットロスのマージン (デフォルト: 0.1) */
  margin?: number;
  /** L2正則化の強さ (デフォルト: 0.001) */
  regularization?: number;
}

/**
 * Contrastive Learning (トリプレットロス) を用いて、相対的な距離感から
 * IntentWeights (行列W と バイアスb) を学習するトレーナークラス。
 *
 * "Anchor" を変換したベクトル A' が、"Negative" よりも "Positive" に
 * 設定されたマージン(Margin)分だけ確実により近づくように重みを更新します。
 */
export class TripletTrainer extends BaseTrainer<TripletExample, IntentWeights> {
  private dimension: number;

  /**
   * TripletTrainer のインスタンスを作成します。
   * @param {number} dimension ベクトルの次元数
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

  protected calculateLoss(
    matrix: Float32Array,
    bias: Float32Array,
    example: TripletExample,
    options?: TripletOnlineOptions,
  ): number {
    const dim = this.dimension;
    const margin = options?.margin ?? 0.1;
    const warpedAnchor = new Float32Array(dim);
    applyAffine(matrix, bias, example.anchor, warpedAnchor, dim);

    const posScore = innerProduct(warpedAnchor, example.positive);
    const negScore = innerProduct(warpedAnchor, example.negative);

    return Math.max(0, margin + negScore - posScore);
  }

  protected adamStep(
    matrix: Float32Array,
    bias: Float32Array,
    mMatrix: Float32Array,
    vMatrix: Float32Array,
    mBias: Float32Array,
    vBias: Float32Array,
    example: TripletExample,
    lr: number,
    reg: number,
    t: number,
    options?: TripletOnlineOptions,
  ): void {
    const dim = this.dimension;
    const margin = options?.margin ?? 0.1;

    const warpedAnchor = new Float32Array(dim);
    applyAffine(matrix, bias, example.anchor, warpedAnchor, dim);

    const posScore = innerProduct(warpedAnchor, example.positive);
    const negScore = innerProduct(warpedAnchor, example.negative);

    const loss = margin + negScore - posScore;

    if (loss > 0) {
      const outputGradients = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        outputGradients[i] = example.negative[i] - example.positive[i];
      }

      this.applyAdamToAffine(
        matrix,
        bias,
        mMatrix,
        vMatrix,
        mBias,
        vBias,
        example.anchor,
        outputGradients,
        lr,
        reg,
        t,
      );
    }
  }

  /**
   * オンライン学習 (フィードバックループ) 用のメソッド。
   * 1つのトリプレットデータからリアルタイムに重みを微調整します。
   *
   * @param {IntentWeights} currentWeights - 現在の重み
   * @param {TripletExample} example - アンカー、正解、不正解を含むトリプレットデータ
   * @param {TripletOnlineOptions} [options={}] - 学習オプション
   * @returns {Promise<IntentWeights>} 微調整された新しい重み
   */
  public async updateOnline(
    currentWeights: IntentWeights,
    example: TripletExample,
    options: TripletOnlineOptions = {},
  ): Promise<IntentWeights> {
    return wasmMutex.runExclusive(async () => {
      this.validateHyperparameters(options);

      const learningRate = options.learningRate ?? 0.01;
      const margin = options.margin ?? 0.1;
      const regularization = options.regularization ?? 0.001;
      assertDimension(
        example.anchor,
        this.dimension,
        "TripletTrainer.train anchor",
      );
      assertDimension(
        example.positive,
        this.dimension,
        "TripletTrainer.train positive",
      );
      assertDimension(
        example.negative,
        this.dimension,
        "TripletTrainer.train negative",
      );

      const dim = this.dimension;
      const { flatMatrix, bias } = getFlatMatrixAndBias(
        currentWeights,
        dim,
        "updateOnline Matrix",
      );

      this.t += 1;
      this.adamStep(
        flatMatrix,
        bias,
        this.mW,
        this.vW,
        this.mb,
        this.vb,
        example,
        learningRate,
        regularization,
        this.t,
        options,
      );

      return this.toWeightsWithRouting(flatMatrix, bias, currentWeights);
    });
  }

  protected override validateHyperparameters(
    options: TripletOnlineOptions,
  ): void {
    super.validateHyperparameters(options);
    if (options.margin !== undefined) {
      if (
        typeof options.margin !== "number" ||
        options.margin < 0 ||
        Number.isNaN(options.margin)
      ) {
        throw new Error("TripletTrainer: margin must be a non-negative number.");
      }
    }
  }
}
