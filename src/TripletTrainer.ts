import { IntentWeights } from "./IntentAdapter";
import { assertDimension, getFlatMatrixAndBias, applyAffine, innerProduct } from "./utils";
import { AbstractAdamTrainer } from "./BaseTrainer";

/**
 * 学習データのペア（Anchor, Positive, Negative）
 * @interface TripletExample
 */
export interface TripletExample {
  /** 基準となるベクトル（検索クエリなど） */
  anchor: number[] | Float32Array;
  /** Anchorに近づけたい正解ベクトル（クリックされた商品など） */
  positive: number[] | Float32Array;
  /** Anchorから遠ざけたい不正解ベクトル（スルーされた商品など） */
  negative: number[] | Float32Array;
}

/**
 * Contrastive Learning (トリプレットロス) を用いて、相対的な距離感から
 * IntentWeights (行列W と バイアスb) を学習するトレーナークラス。
 * 
 * "Anchor" を変換したベクトル A' が、"Negative" よりも "Positive" に
 * 設定されたマージン(Margin)分だけ確実により近づくように重みを更新します。
 */
export class TripletTrainer extends AbstractAdamTrainer {
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

  private toWeights(
    flatMatrix: Float32Array,
    bias: Float32Array,
  ): IntentWeights {
    return {
      matrix: flatMatrix, // 変換のオーバーヘッドを避けるためネイティブ配列のまま返す
      bias: bias,
    };
  }

  /**
   * オンライン学習 (フィードバックループ) 用のメソッド。
   * 1つのトリプレットデータからリアルタイムに重みを微調整します。
   *
   * @param {IntentWeights} currentWeights - 現在の重み
   * @param {number[] | Float32Array} anchor - 検索されたクエリベクトル
   * @param {number[] | Float32Array} positive - 正解（近づけたい）ベクトル
   * @param {number[] | Float32Array} negative - 不正解（遠ざけたい）ベクトル
   * @param {number} learningRate - 1ステップの学習率 (デフォルト: 0.01)
   * @param {number} margin - トリプレットロスのマージン (デフォルト: 0.1)
   * @param {number} regularization - L2正則化の強さ (デフォルト: 0.001)
   * @returns {Promise<IntentWeights>} 微調整された新しい重み
   */
  public async updateOnline(
    currentWeights: IntentWeights,
    anchor: number[] | Float32Array,
    positive: number[] | Float32Array,
    negative: number[] | Float32Array,
    learningRate: number = 0.01,
    margin: number = 0.1,
    regularization: number = 0.001,
  ): Promise<IntentWeights> {
    assertDimension(anchor, this.dimension, "TripletTrainer.train anchor");
    assertDimension(positive, this.dimension, "TripletTrainer.train positive");
    assertDimension(negative, this.dimension, "TripletTrainer.train negative");

    const dim = this.dimension;
    const { flatMatrix, bias } = getFlatMatrixAndBias(currentWeights, dim, "updateOnline Matrix");

    // 1. Forward Pass: アンカーベクトルを現在のアフィン変換でワープさせる A' = W * A + b
    const warpedAnchor = new Float32Array(dim);
    applyAffine(flatMatrix, bias, anchor, warpedAnchor, dim);

    // 2. Compute Loss: L = max(0, margin + (A' * N) - (A' * P))
    // ※内積が大きいほど「近い（類似度が高い）」とみなします。
    const posScore = innerProduct(warpedAnchor, positive);
    const negScore = innerProduct(warpedAnchor, negative);

    const loss = margin + negScore - posScore;

    // 3. Backward Pass: ロスが0より大きい（マージンを満たしていない）場合のみ重みを更新
    if (loss > 0) {
      this.t += 1;
      const beta1 = 0.9;
      const beta2 = 0.999;
      const epsilon = 1e-8;

      // 勾配: dL / dA'_i = N_i - P_i
      for (let i = 0; i < dim; i++) {
        const gradA_i = negative[i] - positive[i];
        
        // バイアスに対する勾配
        const gradB_i = gradA_i;

        // Adam for Bias
        this.mb[i] = beta1 * this.mb[i] + (1 - beta1) * gradB_i;
        this.vb[i] = beta2 * this.vb[i] + (1 - beta2) * (gradB_i * gradB_i);
        const mHatB = this.mb[i] / (1 - Math.pow(beta1, this.t));
        const vHatB = this.vb[i] / (1 - Math.pow(beta2, this.t));
        
        bias[i] -= learningRate * mHatB / (Math.sqrt(vHatB) + epsilon);

        const rowOffset = i * dim;
        for (let j = 0; j < dim; j++) {
          // 行列に対する勾配: dL / dW_ij
          const gradW_ij = gradA_i * anchor[j] + regularization * flatMatrix[rowOffset + j];

          // Adam for Weights
          const idx = rowOffset + j;
          this.mW[idx] = beta1 * this.mW[idx] + (1 - beta1) * gradW_ij;
          this.vW[idx] = beta2 * this.vW[idx] + (1 - beta2) * (gradW_ij * gradW_ij);
          const mHatW = this.mW[idx] / (1 - Math.pow(beta1, this.t));
          const vHatW = this.vW[idx] / (1 - Math.pow(beta2, this.t));

          flatMatrix[idx] -= learningRate * mHatW / (Math.sqrt(vHatW) + epsilon);
        }
      }
    }

    const newWeights = this.toWeights(flatMatrix, bias);
    if (currentWeights.routingVector) {
      newWeights.routingVector = [...currentWeights.routingVector];
    }
    return newWeights;
  }
}
