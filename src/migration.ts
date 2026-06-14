import { ProjectionWeights } from "./ProjectionAdapter";

/**
 * 移行用の学習データペア（旧モデルのベクトル -> 新モデルのベクトル）
 */
export interface MigrationExample {
  source: number[] | Float32Array;
  target: number[] | Float32Array;
}

/**
 * 学習時の最適化オプション
 */
export interface MigrationOptions {
  learningRate?: number;
  epochs?: number;
  regularization?: number;
  momentum?: number;
}

/**
 * 異なる埋め込みモデル間（例：1536次元から512次元へ）の
 * ベクトル空間を翻訳する行列 (ProjectionWeights) を自動学習するトレーナークラス。
 */
export class MigrationTrainer {
  private sourceDimension: number;
  private targetDimension: number;
  private examples: MigrationExample[] = [];

  /**
   * @param {number} sourceDimension 移行元の次元数 (例: ada-002なら1536)
   * @param {number} targetDimension 移行先の次元数 (例: text-embedding-3-smallなら512)
   */
  constructor(sourceDimension: number, targetDimension: number) {
    this.sourceDimension = sourceDimension;
    this.targetDimension = targetDimension;
  }

  public addExample(example: MigrationExample): void {
    if (example.source.length !== this.sourceDimension) {
      throw new Error(
        `Source dimension mismatch. Expected ${this.sourceDimension}.`,
      );
    }
    if (example.target.length !== this.targetDimension) {
      throw new Error(
        `Target dimension mismatch. Expected ${this.targetDimension}.`,
      );
    }
    this.examples.push(example);
  }

  /**
   * サンプルデータをもとに SGD+Momentum で翻訳行列を学習します。
   *
   * @param {MigrationOptions} options
   * @returns {ProjectionWeights} 学習された ProjectionAdapter 用の重み
   */
  public train(options: MigrationOptions = {}): ProjectionWeights {
    if (this.examples.length === 0) {
      throw new Error("No training examples provided.");
    }

    const lr = options.learningRate ?? 0.01;
    const epochs = options.epochs ?? 200;
    const reg = options.regularization ?? 0.001;
    const momentum = options.momentum ?? 0.9;

    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;

    // Projection行列 (tDim 行 x sDim 列)
    const flatMatrix = new Float32Array(tDim * sDim);

    // 非正方行列の初期化 (単位行列の非正方版: x_i -> y_i)
    for (let i = 0; i < tDim; i++) {
      if (i < sDim) {
        flatMatrix[i * sDim + i] = 1.0;
      }
    }

    const bias = new Float32Array(tDim);

    const vMatrix = new Float32Array(tDim * sDim);
    const vBias = new Float32Array(tDim);

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const example of this.examples) {
        this.sgdMomentumStep(
          flatMatrix,
          bias,
          vMatrix,
          vBias,
          example.source,
          example.target,
          lr,
          reg,
          momentum,
        );
      }
    }

    return this.toProjectionWeights(flatMatrix, bias);
  }

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
    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;
    const pred = new Float32Array(tDim);

    // 順伝播: pred = Wx + b
    for (let i = 0; i < tDim; i++) {
      let sum = 0;
      const rowOffset = i * sDim;
      for (let j = 0; j < sDim; j++) {
        sum += matrix[rowOffset + j] * x[j];
      }
      pred[i] = sum + bias[i];
    }

    // 逆伝播 & パラメータ更新
    for (let i = 0; i < tDim; i++) {
      const error = pred[i] - y[i];

      const bGrad = error;
      vBias[i] = momentum * vBias[i] - lr * bGrad;
      bias[i] += vBias[i];

      const rowOffset = i * sDim;
      for (let j = 0; j < sDim; j++) {
        const wIdx = rowOffset + j;
        const wGrad = error * x[j] + reg * matrix[wIdx];
        vMatrix[wIdx] = momentum * vMatrix[wIdx] - lr * wGrad;
        matrix[wIdx] += vMatrix[wIdx];
      }
    }
  }

  private toProjectionWeights(
    flatMatrix: Float32Array,
    bias: Float32Array,
  ): ProjectionWeights {
    const sDim = this.sourceDimension;
    const tDim = this.targetDimension;
    const outMatrix: number[][] = new Array(tDim);
    for (let i = 0; i < tDim; i++) {
      const row = new Array(sDim);
      const rowOffset = i * sDim;
      for (let j = 0; j < sDim; j++) {
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
