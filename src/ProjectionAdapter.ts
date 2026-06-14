/**
 * 次元削減/拡張のための射影行列の重みを定義するインターフェース
 * @interface ProjectionWeights
 */
export interface ProjectionWeights {
  /**
   * 射影変換行列
   * 行数が outDimension、列数が inDimension となります。
   * @type {number[][]}
   */
  matrix: number[][]; // [outDimension][inDimension]

  /**
   * オプションのバイアスベクトル
   * @type {number[] | Float32Array}
   */
  bias?: number[] | Float32Array;
}

/**
 * ProjectionAdapter クラス
 * PCAやSVDなどで事前計算された射影行列を用いて、ベクトルの次元削減（または拡張）を行います。
 */
export class ProjectionAdapter {
  private readonly inDimension: number;
  private readonly outDimension: number;

  // フラット化された射影行列とバイアスを保存
  private readonly matrices: Map<string, Float32Array>;
  private readonly biases: Map<string, Float32Array>;

  /**
   * ProjectionAdapter を初期化します。
   *
   * @constructor
   * @param {number} inDimension - 変換前の入力ベクトルの次元数
   * @param {number} outDimension - 変換後の出力ベクトルの次元数
   * @param {Record<string, ProjectionWeights>} [projections] - 初期化時に追加する射影設定のマップ
   */
  constructor(
    inDimension: number,
    outDimension: number,
    projections?: Record<string, ProjectionWeights>,
  ) {
    this.inDimension = inDimension;
    this.outDimension = outDimension;
    this.matrices = new Map();
    this.biases = new Map();

    if (projections) {
      for (const [name, weights] of Object.entries(projections)) {
        this.addProjection(name, weights);
      }
    }
  }

  /**
   * 実行時に新しい射影設定を動的に追加または更新します。
   *
   * @param {string} name - 追加または更新する射影設定の名前
   * @param {ProjectionWeights} weights - 射影変換行列のデータ
   * @throws {Error} 行列のサイズが指定された次元数と一致しない場合にエラーをスローします。
   * @returns {void}
   */
  public addProjection(name: string, weights: ProjectionWeights): void {
    const { matrix } = weights;

    if (
      matrix.length !== this.outDimension ||
      matrix[0].length !== this.inDimension
    ) {
      throw new Error(
        `Projection '${name}': Matrix must be of size ${this.outDimension}x${this.inDimension}`,
      );
    }

    const flatMatrix = new Float32Array(this.outDimension * this.inDimension);
    for (let i = 0; i < this.outDimension; i++) {
      for (let j = 0; j < this.inDimension; j++) {
        flatMatrix[i * this.inDimension + j] = matrix[i][j];
      }
    }

    this.matrices.set(name, flatMatrix);

    if (weights.bias) {
      if (weights.bias.length !== this.outDimension) {
        throw new Error(
          `Projection '${name}': Bias must be of length ${this.outDimension}`,
        );
      }
      this.biases.set(name, new Float32Array(weights.bias));
    } else {
      this.biases.delete(name);
    }
  }

  /**
   * 指定した射影設定を削除します。
   *
   * @param {string} name - 削除する射影設定の名前
   * @returns {void}
   */
  public removeProjection(name: string): void {
    this.matrices.delete(name);
    this.biases.delete(name);
  }

  /**
   * ベクトルに射影変換を適用し、次元を変更した新しいベクトルを返します。
   * 数式: y = W * x
   *
   * @param {number[] | Float32Array} baseVector - 変換元の入力ベクトル
   * @param {string} projectionName - 適用する射影設定の名前
   * @returns {Float32Array} 射影変換適用後の新しいベクトル
   * @throws {Error} ベクトルの次元数が inDimension と一致しない場合、または射影設定が存在しない場合にエラーをスローします。
   */
  public project(
    baseVector: number[] | Float32Array,
    projectionName: string,
  ): Float32Array {
    if (baseVector.length !== this.inDimension) {
      throw new Error(
        `Vector dimension mismatch. Expected ${this.inDimension}, got ${baseVector.length}.`,
      );
    }

    const matrix = this.matrices.get(projectionName);
    if (!matrix) {
      throw new Error(`Projection '${projectionName}' not found.`);
    }

    const result = new Float32Array(this.outDimension);
    const bias = this.biases.get(projectionName);

    // 行列ベクトル積: O(M * N)
    for (let i = 0; i < this.outDimension; i++) {
      let sum = bias ? bias[i] : 0;
      const rowOffset = i * this.inDimension;
      for (let j = 0; j < this.inDimension; j++) {
        sum += matrix[rowOffset + j] * baseVector[j];
      }
      result[i] = sum;
    }

    return result;
  }
}
