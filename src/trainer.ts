import { IntentWeights } from "./IntentAdapter";
import { BaseTrainer, BaseTrainingOptions } from "./BaseTrainer";
import { initWasm } from "./wasm/wasm-loader";
import { flattenMatrix } from "./utils";

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
 */
export interface TrainingOptions extends BaseTrainingOptions {}

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

  protected toWeights(
    flatMatrix: Float32Array,
    bias: Float32Array,
  ): IntentWeights {
    const dim = this.dimension;
    const outMatrix: number[][] = new Array(dim);
    for (let i = 0; i < dim; i++) {
      const row = new Array(dim);
      const rowOffset = i * dim;
      for (let j = 0; j < dim; j++) {
        row[j] = flatMatrix[rowOffset + j];
      }
      outMatrix[i] = row;
    }
    return {
      matrix: outMatrix,
      bias: Array.from(bias),
    };
  }

  /**
   * オンライン学習 (フィードバックループ) 用のメソッド。
   * ユーザーのクリックなどの 1 回のフィードバックからリアルタイムに重みを微調整します。
   *
   * @param {IntentWeights} currentWeights - 現在の重み
   * @param {number[] | Float32Array} input - 検索されたクエリベクトル
   * @param {number[] | Float32Array} target - クリックされた(理想の)ドキュメントのベクトル
   * @param {number} learningRate - 1ステップの学習率 (デフォルト: 0.01)
   * @param {number} regularization - L2正則化の強さ (デフォルト: 0.001)
   * @returns {IntentWeights} 微調整された新しい重み
   */
  public async updateOnline(
    currentWeights: IntentWeights,
    input: number[] | Float32Array,
    target: number[] | Float32Array,
    learningRate: number = 0.01,
    regularization: number = 0.001,
  ): Promise<IntentWeights> {
    await initWasm();

    if (input.length !== this.dimension || target.length !== this.dimension) {
      throw new Error(`Dimension mismatch. Expected ${this.dimension}`);
    }

    const dim = this.dimension;
    let flatMatrix: Float32Array;
    if (currentWeights.matrix instanceof Float32Array) {
      flatMatrix = new Float32Array(currentWeights.matrix);
    } else {
      flatMatrix = flattenMatrix(currentWeights.matrix, dim, dim, "updateOnline Matrix");
    }
    const bias = new Float32Array(currentWeights.bias);

    // オンライン更新では Momentum を 0 として1ステップのみのSGDを行う
    const vMatrix = new Float32Array(dim * dim);
    const vBias = new Float32Array(dim);

    this.sgdMomentumStep(
      flatMatrix,
      bias,
      vMatrix,
      vBias,
      input,
      target,
      learningRate,
      regularization,
      0.0, // no momentum for 1-shot online update
    );

    const newWeights = this.toWeights(flatMatrix, bias);
    if (currentWeights.routingVector) {
      newWeights.routingVector = [...currentWeights.routingVector];
    }
    return newWeights;
  }
}
