import { IntentWeights, initWasm, wasmMutex } from "@warpvector/core";
import {
  assertDimension,
  getFlatMatrixAndBias,
  applyAffine,
  innerProduct,
  softmax,
} from "@warpvector/core";
import { BaseTrainer } from "../trainers/BaseTrainer";

/**
 * 学習データのペア（Anchor, Positive, 複数のNegatives）
 * @interface InfoNCEExample
 */
export interface InfoNCEExample {
  /** 基準となるベクトル（検索クエリなど） */
  anchor: number[] | Float32Array;
  /** Anchorに近づけたい正解ベクトル（クリックされた商品など） */
  positive: number[] | Float32Array;
  /** Anchorから遠ざけたい不正解ベクトルの配列（スルーされた商品群など） */
  negatives: (number[] | Float32Array)[];
}

/**
 * InfoNCETrainer のオンライン学習オプション
 */
export interface InfoNCEOnlineOptions {
  /** 1ステップの学習率 (デフォルト: 0.01) */
  learningRate?: number;
  /** Softmaxの温度パラメータ (デフォルト: 0.1) */
  temperature?: number;
  /** L2正則化の強さ (デフォルト: 0.001) */
  regularization?: number;
}

/**
 * InfoNCE Loss (Softmax Cross Entropy) を用いて、
 * 「1つの正解を近づけ、複数の不正解を一気に遠ざける」ように
 * IntentWeights (行列W と バイアスb) を学習するトレーナークラス。
 */
export class InfoNCETrainer extends BaseTrainer<InfoNCEExample, IntentWeights> {
  private dimension: number;

  /**
   * InfoNCETrainer のインスタンスを作成します。
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
    example: InfoNCEExample,
    options?: InfoNCEOnlineOptions,
  ): number {
    const dim = this.dimension;
    const temperature = options?.temperature ?? 0.1;

    const warpedAnchor = new Float32Array(dim);
    applyAffine(matrix, bias, example.anchor, warpedAnchor, dim);

    const posScore = innerProduct(warpedAnchor, example.positive);
    const numNegatives = example.negatives.length;

    const allScores = [posScore / temperature];
    for (let n = 0; n < numNegatives; n++) {
      allScores.push(
        innerProduct(warpedAnchor, example.negatives[n]) / temperature,
      );
    }

    const probs = softmax(allScores);
    const pPos = probs[0];

    // Cross entropy loss for the positive class
    return -Math.log(Math.max(pPos, 1e-9));
  }

  protected adamStep(
    matrix: Float32Array,
    bias: Float32Array,
    mMatrix: Float32Array,
    vMatrix: Float32Array,
    mBias: Float32Array,
    vBias: Float32Array,
    example: InfoNCEExample,
    lr: number,
    reg: number,
    t: number,
    options?: InfoNCEOnlineOptions,
  ): void {
    const dim = this.dimension;
    const temperature = options?.temperature ?? 0.1;

    const warpedAnchor = new Float32Array(dim);
    applyAffine(matrix, bias, example.anchor, warpedAnchor, dim);

    const posScore = innerProduct(warpedAnchor, example.positive);
    const numNegatives = example.negatives.length;

    const allScores = [posScore / temperature];
    for (let n = 0; n < numNegatives; n++) {
      allScores.push(
        innerProduct(warpedAnchor, example.negatives[n]) / temperature,
      );
    }

    const probs = softmax(allScores);
    const pPos = probs[0];

    const outputGradients = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      let gradA_i = (pPos - 1.0) * example.positive[i];
      for (let n = 0; n < numNegatives; n++) {
        gradA_i += probs[n + 1] * example.negatives[n][i];
      }
      outputGradients[i] = gradA_i / temperature;
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

  /**
   * オンライン学習 (フィードバックループ) 用のメソッド。
   * 1つのクエリ(Anchor)、1つのクリック(Positive)、複数のスルー(Negatives)から重みを微調整します。
   *
   * @param {IntentWeights} currentWeights - 現在の重み
   * @param {InfoNCEExample} example - アンカー、正解、複数の不正解を含むデータ
   * @param {InfoNCEOnlineOptions} [options={}] - 学習オプション
   * @returns {Promise<IntentWeights>} 微調整された新しい重み
   */
  public async updateOnline(
    currentWeights: IntentWeights,
    example: InfoNCEExample,
    options: InfoNCEOnlineOptions = {},
  ): Promise<IntentWeights> {
    return wasmMutex.runExclusive(async () => {
      this.validateHyperparameters(options);

      const learningRate = options.learningRate ?? 0.01;
      const regularization = options.regularization ?? 0.001;
      const dim = this.dimension;

      assertDimension(example.anchor, dim, "InfoNCETrainer.train anchor");
      assertDimension(example.positive, dim, "InfoNCETrainer.train positive");
      if (example.negatives.length === 0) {
        throw new Error("InfoNCETrainer requires at least one negative example.");
      }
      for (const neg of example.negatives) {
        assertDimension(neg, dim, "InfoNCETrainer.train negative");
      }

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
    options: InfoNCEOnlineOptions,
  ): void {
    super.validateHyperparameters(options);
    if (options.temperature !== undefined) {
      if (
        typeof options.temperature !== "number" ||
        options.temperature <= 0 ||
        Number.isNaN(options.temperature)
      ) {
        throw new Error("InfoNCETrainer: temperature must be a positive number.");
      }
    }
  }
}
