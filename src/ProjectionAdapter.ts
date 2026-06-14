export interface ProjectionWeights {
  matrix: number[][]; // [outDimension][inDimension]
}

export class ProjectionAdapter {
  private readonly inDimension: number;
  private readonly outDimension: number;
  
  // フラット化された射影行列を保存
  private readonly matrices: Map<string, Float32Array>;

  /**
   * ProjectionAdapter を初期化します。
   * PCAなどにより事前計算された射影行列を用いて、ベクトルの次元削減（または拡張）を行います。
   *
   * @param inDimension 入力ベクトルの次元数
   * @param outDimension 出力ベクトルの次元数
   * @param projections （オプション）初期化時に追加する射影行列
   */
  constructor(inDimension: number, outDimension: number, projections?: Record<string, ProjectionWeights>) {
    this.inDimension = inDimension;
    this.outDimension = outDimension;
    this.matrices = new Map();

    if (projections) {
      for (const [name, weights] of Object.entries(projections)) {
        this.addProjection(name, weights);
      }
    }
  }

  public addProjection(name: string, weights: ProjectionWeights): void {
    const { matrix } = weights;

    if (matrix.length !== this.outDimension || matrix[0].length !== this.inDimension) {
      throw new Error(`Projection '${name}': Matrix must be of size ${this.outDimension}x${this.inDimension}`);
    }

    const flatMatrix = new Float32Array(this.outDimension * this.inDimension);
    for (let i = 0; i < this.outDimension; i++) {
      for (let j = 0; j < this.inDimension; j++) {
        flatMatrix[i * this.inDimension + j] = matrix[i][j];
      }
    }

    this.matrices.set(name, flatMatrix);
  }

  public removeProjection(name: string): void {
    this.matrices.delete(name);
  }

  /**
   * ベクトルに射影変換を適用し、次元を変更した新しいベクトルを返します。
   * y = W * x
   * 
   * @param baseVector 変換元のベクトル
   * @param projectionName 適用する射影設定の名前
   */
  public project(baseVector: number[] | Float32Array, projectionName: string): Float32Array {
    if (baseVector.length !== this.inDimension) {
      throw new Error(`Vector dimension mismatch. Expected ${this.inDimension}, got ${baseVector.length}.`);
    }

    const matrix = this.matrices.get(projectionName);
    if (!matrix) {
      throw new Error(`Projection '${projectionName}' not found.`);
    }

    const result = new Float32Array(this.outDimension);

    // 行列ベクトル積: O(M * N)
    for (let i = 0; i < this.outDimension; i++) {
      let sum = 0;
      const rowOffset = i * this.inDimension;
      for (let j = 0; j < this.inDimension; j++) {
        sum += matrix[rowOffset + j] * baseVector[j];
      }
      result[i] = sum;
    }

    return result;
  }
}
