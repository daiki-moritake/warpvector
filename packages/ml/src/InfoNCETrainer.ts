import { IntentWeights } from "@warpvector/core";
import {
  assertDimension,
  getFlatMatrixAndBias,
  applyAffine,
  innerProduct,
} from "@warpvector/core";
import { AbstractAdamTrainer } from "./BaseTrainer";

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
export class InfoNCETrainer extends AbstractAdamTrainer {
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

  private toWeights(
    flatMatrix: Float32Array,
    bias: Float32Array,
  ): IntentWeights {
    return {
      matrix: flatMatrix, // ネイティブ配列のまま返す
      bias: bias,
    };
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
    const learningRate = options.learningRate ?? 0.01;
    const temperature = options.temperature ?? 0.1;
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

    // 1. Forward Pass: アンカーベクトルを現在のアフィン変換でワープさせる A' = W * A + b
    const warpedAnchor = new Float32Array(dim);
    applyAffine(flatMatrix, bias, example.anchor, warpedAnchor, dim);

    // 2. スコア計算: s(A', X) = A' \cdot X
    const posScore = innerProduct(warpedAnchor, example.positive);

    const negScores = new Float32Array(example.negatives.length);
    for (let n = 0; n < example.negatives.length; n++) {
      negScores[n] = innerProduct(warpedAnchor, example.negatives[n]);
    }

    // 3. Softmax 確率の計算 (数値的安定性のために max を引く)
    let maxScore = posScore / temperature;
    for (let n = 0; n < example.negatives.length; n++) {
      const s = negScores[n] / temperature;
      if (s > maxScore) maxScore = s;
    }

    const expPos = Math.exp(posScore / temperature - maxScore);
    const expNegs = new Float32Array(example.negatives.length);
    let sumExp = expPos;

    for (let n = 0; n < example.negatives.length; n++) {
      const expN = Math.exp(negScores[n] / temperature - maxScore);
      expNegs[n] = expN;
      sumExp += expN;
    }

    const pPos = expPos / sumExp; // 正解の予測確率
    const pNegs = new Float32Array(example.negatives.length);
    for (let n = 0; n < example.negatives.length; n++) {
      pNegs[n] = expNegs[n] / sumExp; // 各不正解の予測確率
    }

    // 4. Backward Pass: 勾配計算と Adam Optimizer
    this.t += 1;
    // dL/dA'_i = (1 / tau) * [ (pPos - 1) * P_i + sum_k (pNegs_k * N_ki) ]
    const outputGradients = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      let gradA_i = (pPos - 1.0) * example.positive[i];
      for (let n = 0; n < example.negatives.length; n++) {
        gradA_i += pNegs[n] * example.negatives[n][i];
      }
      outputGradients[i] = gradA_i / temperature;
    }

    this.applyAdamToAffine(
      flatMatrix,
      bias,
      this.mW,
      this.vW,
      this.mb,
      this.vb,
      example.anchor,
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
