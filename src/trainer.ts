import { IntentWeights } from "./IntentAdapter";

/**
 * 学習データのペア（ベースのクエリベクトルと、目標となる結果ベクトル）
 */
export interface TrainingExample {
  input: number[] | Float32Array;
  target: number[] | Float32Array;
}

/**
 * 学習時の最適化オプション
 */
export interface TrainingOptions {
  /** 学習率 (デフォルト: 0.01) */
  learningRate?: number;
  /** 学習エポック数 (デフォルト: 100) */
  epochs?: number;
  /** L2正則化の強さ (Ridge回帰). 極端な重みの増大を防ぐ (デフォルト: 0.001) */
  regularization?: number;
  /** Momentum (慣性項). 収束を早め、局所解を回避する (デフォルト: 0.9) */
  momentum?: number;
}

/**
 * サンプルデータから最適な `IntentWeights` (行列Wとバイアスb) を
 * 確率的勾配降下法 (SGD + Momentum) により自動学習するトレーナークラス。
 * 内部で Float32Array のフラット化を行い、極限のパフォーマンスを引き出します。
 */
export class IntentTrainer {
  private dimension: number;
  private examples: TrainingExample[] = [];

  constructor(dimension: number) {
    this.dimension = dimension;
  }

  /**
   * 学習用のデータ（正例）を追加します。
   *
   * @param {TrainingExample} example
   */
  public addExample(example: TrainingExample): void {
    if (
      example.input.length !== this.dimension ||
      example.target.length !== this.dimension
    ) {
      throw new Error(`Dimension mismatch. Expected ${this.dimension}`);
    }
    this.examples.push(example);
  }

  /**
   * 現在のサンプルデータをもとに、最適な IntentWeights を学習します。
   * 内部でフラットな Float32Array に変換し、キャッシュ局所性を高めて高速化します。
   *
   * @param {TrainingOptions} [options]
   * @returns {IntentWeights} 学習された変換行列とバイアス
   */
  public train(options: TrainingOptions = {}): IntentWeights {
    if (this.examples.length === 0) {
      throw new Error("No training examples provided.");
    }

    const lr = options.learningRate ?? 0.01;
    const epochs = options.epochs ?? 100;
    const reg = options.regularization ?? 0.001;
    const momentum = options.momentum ?? 0.9;

    const dim = this.dimension;

    // 行列 W は初期状態を単位行列 (Identity Matrix) に設定
    const flatMatrix = new Float32Array(dim * dim);
    for (let i = 0; i < dim; i++) {
      flatMatrix[i * dim + i] = 1.0;
    }
    const bias = new Float32Array(dim);

    // Momentum最適化のための速度(Velocity)ベクトル
    const vMatrix = new Float32Array(dim * dim);
    const vBias = new Float32Array(dim);

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const example of this.examples) {
        this.sgdMomentumStep(
          flatMatrix,
          bias,
          vMatrix,
          vBias,
          example.input,
          example.target,
          lr,
          reg,
          momentum,
        );
      }
    }

    // 出力用の number[][] 形式に復元する
    return this.toIntentWeights(flatMatrix, bias);
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
  public updateOnline(
    currentWeights: IntentWeights,
    input: number[] | Float32Array,
    target: number[] | Float32Array,
    learningRate: number = 0.01,
    regularization: number = 0.001,
  ): IntentWeights {
    if (input.length !== this.dimension || target.length !== this.dimension) {
      throw new Error(`Dimension mismatch. Expected ${this.dimension}`);
    }

    const dim = this.dimension;
    const flatMatrix = new Float32Array(dim * dim);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        flatMatrix[i * dim + j] = currentWeights.matrix[i][j];
      }
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

    const newWeights = this.toIntentWeights(flatMatrix, bias);
    if (currentWeights.routingVector) {
      newWeights.routingVector = [...currentWeights.routingVector];
    }
    return newWeights;
  }

  /**
   * 1ステップの勾配降下法 (SGD + Momentum) を実行し、Wとbをインプレース更新する
   */
  private sgdMomentumStep(
    matrix: Float32Array,
    bias: Float32Array,
    vMatrix: Float32Array,
    vBias: Float32Array,
    x: number[] | Float32Array,
    y: number[] | Float32Array,
    lr: number,
    reg: number,
    momentum: number,
  ): void {
    const dim = this.dimension;
    const pred = new Float32Array(dim);

    // 順伝播: pred = Wx + b
    for (let i = 0; i < dim; i++) {
      let sum = 0;
      const rowOffset = i * dim;
      for (let j = 0; j < dim; j++) {
        sum += matrix[rowOffset + j] * x[j];
      }
      pred[i] = sum + bias[i];
    }

    // 逆伝播 & パラメータ更新 (Momentum を考慮)
    // 誤差 error = pred - y
    for (let i = 0; i < dim; i++) {
      const error = pred[i] - y[i];

      // バイアスの更新: v_b = momentum * v_b - lr * dL/db
      const bGrad = error;
      vBias[i] = momentum * vBias[i] - lr * bGrad;
      bias[i] += vBias[i];

      const rowOffset = i * dim;
      for (let j = 0; j < dim; j++) {
        // 行列の更新: v_W = momentum * v_W - lr * (dL/dW + L2)
        const wIdx = rowOffset + j;
        const wGrad = error * x[j] + reg * matrix[wIdx];
        vMatrix[wIdx] = momentum * vMatrix[wIdx] - lr * wGrad;
        matrix[wIdx] += vMatrix[wIdx];
      }
    }
  }

  /**
   * 内部の Float32Array を IntentAdapter が受け取れる number[][] 形式に変換する
   */
  private toIntentWeights(
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
}
