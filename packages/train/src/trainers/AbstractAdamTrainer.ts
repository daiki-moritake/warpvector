import { assertDimension } from "@warpvector/core";
import { applyAdamToAffine } from "../optimizers/adam";

/**
 * Adam最適化のステート変数を管理する共通基底クラス
 */
export abstract class AbstractAdamTrainer<
  TResult = import("@warpvector/core").IntentWeights,
> {
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

  /**
   * 学習済みの1次元行列とバイアスから最終的な重みデータを構築します。
   * デフォルトでは IntentWeights または互換性のある型にキャストして返します。
   * @param flatMatrix 学習済みのフラット化された行列
   * @param bias 学習済みのバイアスベクトル
   * @returns TResult
   */
  protected toWeights(flatMatrix: Float32Array, bias: Float32Array): TResult {
    return {
      matrix: flatMatrix,
      bias: bias,
    } as unknown as TResult;
  }

  /**
   * 学習済みの行列とバイアスを IntentWeights に変換し、
   * 必要に応じて元の routingVector を引き継ぎます。
   *
   * @param flatMatrix 学習済みのフラット化された行列
   * @param bias 学習済みのバイアスベクトル
   * @param currentWeights 現在の重み（routingVectorを引き継ぐため）
   * @returns routingVectorが引き継がれた新しい重み
   */
  protected toWeightsWithRouting(
    flatMatrix: Float32Array,
    bias: Float32Array,
    currentWeights: import("@warpvector/core").IntentWeights,
  ): import("@warpvector/core").IntentWeights {
    const newWeights = this.toWeights(
      flatMatrix,
      bias,
    ) as unknown as import("@warpvector/core").IntentWeights;
    if (currentWeights.routingVector) {
      newWeights.routingVector = [...currentWeights.routingVector];
    }
    return newWeights;
  }

  /**
   * アフィンレイヤー (matrix, bias) に対する Adam のパラメータ更新を適用します。
   * InfoNCE や Triplet など、様々な損失関数で計算された勾配(outputGradients)を元に更新を行います。
   */
  protected applyAdamToAffine(
    matrix: Float32Array,
    bias: Float32Array,
    mMatrix: Float32Array,
    vMatrix: Float32Array,
    mBias: Float32Array,
    vBias: Float32Array,
    input: number[] | Float32Array,
    outputGradients: number[] | Float32Array,
    lr: number,
    reg: number,
    t: number,
  ): void {
    applyAdamToAffine(
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
      t,
    );
  }
}
