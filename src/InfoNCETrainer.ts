import { IntentWeights } from "./IntentAdapter";
import { flattenMatrix, assertDimension } from "./utils";
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
   * @param {number[] | Float32Array} anchor - 検索されたクエリベクトル
   * @param {number[] | Float32Array} positive - 正解（近づけたい）ベクトル
   * @param {(number[] | Float32Array)[]} negatives - 不正解（遠ざけたい）ベクトルの配列
   * @param {number} learningRate - 1ステップの学習率 (デフォルト: 0.01)
   * @param {number} temperature - Softmaxの温度パラメータ (デフォルト: 0.1)
   * @param {number} regularization - L2正則化の強さ (デフォルト: 0.001)
   * @returns {Promise<IntentWeights>} 微調整された新しい重み
   */
  public async updateOnline(
    currentWeights: IntentWeights,
    anchor: number[] | Float32Array,
    positive: number[] | Float32Array,
    negatives: (number[] | Float32Array)[],
    learningRate: number = 0.01,
    temperature: number = 0.1,
    regularization: number = 0.001,
  ): Promise<IntentWeights> {
    const dim = this.dimension;

    assertDimension(anchor, dim, "InfoNCETrainer.train anchor");
    assertDimension(positive, dim, "InfoNCETrainer.train positive");
    for (let i = 0; i < negatives.length; i++) {
      assertDimension(negatives[i], dim, `InfoNCETrainer.train negative[${i}]`);
    }

    let flatMatrix: Float32Array;
    if (currentWeights.matrix instanceof Float32Array) {
      flatMatrix = new Float32Array(currentWeights.matrix);
    } else {
      flatMatrix = flattenMatrix(currentWeights.matrix, dim, dim, "updateOnline Matrix");
    }
    const bias = new Float32Array(currentWeights.bias);

    // 1. Forward Pass: アンカーベクトルを現在のアフィン変換でワープさせる A' = W * A + b
    const warpedAnchor = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      let sum = bias[i];
      const rowOffset = i * dim;
      for (let j = 0; j < dim; j++) {
        sum += flatMatrix[rowOffset + j] * anchor[j];
      }
      warpedAnchor[i] = sum;
    }

    // 2. スコア計算: s(A', X) = A' \cdot X
    let posScore = 0;
    for (let i = 0; i < dim; i++) {
      posScore += warpedAnchor[i] * positive[i];
    }

    const negScores = new Float32Array(negatives.length);
    for (let n = 0; n < negatives.length; n++) {
      let score = 0;
      for (let i = 0; i < dim; i++) {
        score += warpedAnchor[i] * negatives[n][i];
      }
      negScores[n] = score;
    }

    // 3. Softmax 確率の計算 (数値的安定性のために max を引く)
    let maxScore = posScore / temperature;
    for (let n = 0; n < negatives.length; n++) {
      const s = negScores[n] / temperature;
      if (s > maxScore) maxScore = s;
    }

    const expPos = Math.exp(posScore / temperature - maxScore);
    const expNegs = new Float32Array(negatives.length);
    let sumExp = expPos;

    for (let n = 0; n < negatives.length; n++) {
      const expN = Math.exp(negScores[n] / temperature - maxScore);
      expNegs[n] = expN;
      sumExp += expN;
    }

    const pPos = expPos / sumExp; // 正解の予測確率
    const pNegs = new Float32Array(negatives.length);
    for (let n = 0; n < negatives.length; n++) {
      pNegs[n] = expNegs[n] / sumExp; // 各不正解の予測確率
    }

    // 4. Backward Pass: 勾配計算と Adam Optimizer
    this.t += 1;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    // dL/dA'_i = (1 / tau) * [ (pPos - 1) * P_i + sum_k (pNegs_k * N_ki) ]
    for (let i = 0; i < dim; i++) {
      let gradA_i = (pPos - 1.0) * positive[i];
      for (let n = 0; n < negatives.length; n++) {
        gradA_i += pNegs[n] * negatives[n][i];
      }
      gradA_i /= temperature;

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

    const newWeights = this.toWeights(flatMatrix, bias);
    if (currentWeights.routingVector) {
      newWeights.routingVector = [...currentWeights.routingVector];
    }
    return newWeights;
  }
}
