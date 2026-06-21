import { ProjectionWeights, applyAffine } from "@warpvector/core";
import { BaseTrainer, BaseTrainingOptions } from "./BaseTrainer";

/**
 * Cross-Encoder のための学習データペア
 * クエリとドキュメントのペアに対して、関連度スコア（例: 0.0 ~ 1.0）を与えます。
 */
export interface CrossEncoderExample {
  /** クエリベクトル */
  query: number[] | Float32Array;
  /** ドキュメントベクトル */
  document: number[] | Float32Array;
  /** クエリとドキュメントの関連度スコア（ラベル） */
  score: number;
}

/**
 * Cross-Encoder モデル（ペアワイズのスコアリング）の知識を蒸留するためのトレーナー。
 * クエリベクトルとドキュメントベクトルを連結（Concatenate）し、
 * 単一のスカラー値（スコア）を出力する射影行列（ProjectionWeights）を学習します。
 *
 * @example
 * const trainer = new CrossEncoderTrainer(1536);
 * trainer.addExample({ query: [...], document: [...], score: 0.95 });
 * const weights = await trainer.train({ epochs: 100, learningRate: 0.01 });
 */
export class CrossEncoderTrainer extends BaseTrainer<
  CrossEncoderExample,
  ProjectionWeights
> {
  private _queryDimension: number;
  private _docDimension: number;

  /**
   * @param queryDimension クエリベクトルの次元数
   * @param docDimension ドキュメントベクトルの次元数（デフォルトは queryDimension と同じ）
   */
  constructor(queryDimension: number, docDimension: number = queryDimension) {
    super();
    this._queryDimension = queryDimension;
    this._docDimension = docDimension;
  }

  protected get sourceDimension(): number {
    // クエリとドキュメントを連結したものが入力次元となる
    return this._queryDimension + this._docDimension;
  }

  protected get targetDimension(): number {
    // 出力は単一のスコア
    return 1;
  }

  /**
   * クエリとドキュメントを連結した特徴ベクトルを作成します。
   */
  private buildInteractionFeature(
    query: number[] | Float32Array,
    document: number[] | Float32Array
  ): Float32Array {
    const feature = new Float32Array(this.sourceDimension);
    for (let i = 0; i < this._queryDimension; i++) {
      feature[i] = query[i];
    }
    for (let i = 0; i < this._docDimension; i++) {
      feature[this._queryDimension + i] = document[i];
    }
    return feature;
  }

  protected calculateLoss(
    matrix: Float32Array,
    bias: Float32Array,
    example: CrossEncoderExample,
    options?: BaseTrainingOptions
  ): number {
    const input = this.buildInteractionFeature(example.query, example.document);
    const pred = new Float32Array(1);
    
    // 順伝播
    applyAffine(matrix, bias, input, pred, this.sourceDimension, 1);
    
    // MSE (Mean Squared Error)
    const diff = pred[0] - example.score;
    return diff * diff;
  }

  protected adamStep(
    matrix: Float32Array,
    bias: Float32Array,
    mMatrix: Float32Array,
    vMatrix: Float32Array,
    mBias: Float32Array,
    vBias: Float32Array,
    example: CrossEncoderExample,
    lr: number,
    reg: number,
    t: number,
    options?: BaseTrainingOptions
  ): void {
    const input = this.buildInteractionFeature(example.query, example.document);
    const pred = new Float32Array(1);
    
    // 順伝播
    applyAffine(matrix, bias, input, pred, this.sourceDimension, 1);
    
    // MSE Loss の微分 (pred - target)
    // ※ 定数係数2は学習率に吸収させるため省略
    const outputGradients = new Float32Array(1);
    outputGradients[0] = pred[0] - example.score;

    // Adamによる逆伝播とパラメータ更新
    this.applyAdamToAffine(
      matrix,
      bias,
      mMatrix,
      vMatrix,
      mBias,
      vBias,
      input,
      outputGradients,
      lr,
      reg,
      t
    );
  }
}
