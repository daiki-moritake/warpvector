export interface IntentWeights {
  matrix: number[][];
  bias: number[];
}

export class IntentAdapter {
  private readonly dimension: number;
  private readonly matrices: Map<string, Float32Array>;
  private readonly biases: Map<string, Float32Array>;

  /**
   * IntentAdapter を初期化します。
   * 行列とバイアスを Float32Array にコンパイルし、キャッシュ局所性と計算速度を最適化します。
   *
   * @param intents 各インテント名と IntentWeights (行列・バイアス) のマッピング
   */
  constructor(intents: Record<string, IntentWeights>) {
    this.matrices = new Map();
    this.biases = new Map();

    const intentKeys = Object.keys(intents);
    if (intentKeys.length === 0) {
      throw new Error("At least one intent must be provided.");
    }

    // 最初のインテントのバイアス長からベクトル空間の次元数を決定
    const firstIntent = intents[intentKeys[0]];
    this.dimension = firstIntent.bias.length;

    for (const [intentName, weights] of Object.entries(intents)) {
      this.addIntent(intentName, weights);
    }
  }

  /**
   * 実行時に新しい意図を動的に追加または更新します。
   *
   * @param intentName 追加する意図の名前
   * @param weights 意図の重み（行列とバイアス）
   */
  public addIntent(intentName: string, weights: IntentWeights): void {
    const { matrix, bias } = weights;

    // バイアスの次元数バリデーション
    if (bias.length !== this.dimension) {
      throw new Error(
        `Intent '${intentName}': Bias dimension mismatch. Expected ${this.dimension}, got ${bias.length}.`
      );
    }

    // 行列の行数バリデーション
    if (matrix.length !== this.dimension) {
      throw new Error(
        `Intent '${intentName}': Matrix row dimension mismatch. Expected ${this.dimension}, got ${matrix.length}.`
      );
    }

    // キャッシュ効率を上げるため、2次元配列をフラットな Float32Array にプリコンパイル
    const flatMatrix = new Float32Array(this.dimension * this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      // 行列の列数バリデーション
      if (matrix[i].length !== this.dimension) {
        throw new Error(
          `Intent '${intentName}': Matrix column dimension mismatch at row ${i}. Expected ${this.dimension}, got ${matrix[i].length}.`
        );
      }
      
      for (let j = 0; j < this.dimension; j++) {
        flatMatrix[i * this.dimension + j] = matrix[i][j];
      }
    }

    this.matrices.set(intentName, flatMatrix);
    this.biases.set(intentName, new Float32Array(bias));
  }

  /**
   * 指定した意図を削除します。
   *
   * @param intentName 削除する意図の名前
   */
  public removeIntent(intentName: string): void {
    this.matrices.delete(intentName);
    this.biases.delete(intentName);
  }

  /**
   * 指定された意図（intent）に基づいて、ベースベクトルにアフィン変換を適用します。
   * x' = W_I * x + b_I
   *
   * @param baseVector 変換元のベクトル (number[] または Float32Array)
   * @param intent 適用する意図（intent）の名前
   * @returns 変換後のベクトルを表す新しい Float32Array
   */
  public tune(baseVector: number[] | Float32Array, intent: string): Float32Array {
    if (baseVector.length !== this.dimension) {
      throw new Error(
        `Vector dimension mismatch. Expected ${this.dimension}, got ${baseVector.length}.`
      );
    }

    const matrix = this.matrices.get(intent);
    const bias = this.biases.get(intent);

    if (!matrix || !bias) {
      throw new Error(`Intent '${intent}' not found.`);
    }

    const dim = this.dimension;
    const result = new Float32Array(dim);

    // 行列・ベクトル積およびバイアス加算
    // メモリアクセスの局所性を最大化するため、行→列の順にループ処理
    for (let i = 0; i < dim; i++) {
      let sum = 0;
      const rowOffset = i * dim;
      
      for (let j = 0; j < dim; j++) {
        sum += matrix[rowOffset + j] * baseVector[j];
      }
      
      result[i] = sum + bias[i];
    }

    return result;
  }

  /**
   * 複数のベースベクトルに対して、指定された意図（intent）のアフィン変換をバッチ処理で適用します。
   * ループのオーバーヘッドを減らし、大規模なデータセットに対してより効率的に変換を行います。
   *
   * @param baseVectors 変換元のベクトルの配列 (2次元配列)
   * @param intent 適用する意図（intent）の名前
   * @returns 変換後のベクトルを格納した Float32Array の配列
   */
  public tuneBatch(baseVectors: (number[] | Float32Array)[], intent: string): Float32Array[] {
    const matrix = this.matrices.get(intent);
    const bias = this.biases.get(intent);

    if (!matrix || !bias) {
      throw new Error(`Intent '${intent}' not found.`);
    }

    const dim = this.dimension;
    const batchSize = baseVectors.length;
    const results = new Array<Float32Array>(batchSize);

    for (let k = 0; k < batchSize; k++) {
      const baseVector = baseVectors[k];
      if (baseVector.length !== dim) {
        throw new Error(
          `Vector dimension mismatch at index ${k}. Expected ${dim}, got ${baseVector.length}.`
        );
      }

      const result = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        let sum = 0;
        const rowOffset = i * dim;
        
        for (let j = 0; j < dim; j++) {
          sum += matrix[rowOffset + j] * baseVector[j];
        }
        
        result[i] = sum + bias[i];
      }
      results[k] = result;
    }

    return results;
  }

  /**
   * 複数の意図を指定された重みでブレンドし、ベクトルにアフィン変換を適用します。
   * W_blend = Σ(w_i * W_i), b_blend = Σ(w_i * b_i)
   *
   * @param baseVector 変換元のベクトル (number[] または Float32Array)
   * @param blendWeights 意図の名前と重みのマッピング (例: { riskAnalysis: 0.7, economicImpact: 0.3 })
   * @returns 変換後のベクトルを表す Float32Array
   */
  public tuneBlended(baseVector: number[] | Float32Array, blendWeights: Record<string, number>): Float32Array {
    if (baseVector.length !== this.dimension) {
      throw new Error(
        `Vector dimension mismatch. Expected ${this.dimension}, got ${baseVector.length}.`
      );
    }

    const dim = this.dimension;
    const blendedMatrix = new Float32Array(dim * dim);
    const blendedBias = new Float32Array(dim);

    // ブレンド行列とバイアスの合成
    for (const [intentName, weight] of Object.entries(blendWeights)) {
      const matrix = this.matrices.get(intentName);
      const bias = this.biases.get(intentName);

      if (!matrix || !bias) {
        throw new Error(`Intent '${intentName}' not found during blending.`);
      }

      for (let i = 0; i < dim; i++) {
        blendedBias[i] += bias[i] * weight;
        const rowOffset = i * dim;
        for (let j = 0; j < dim; j++) {
          blendedMatrix[rowOffset + j] += matrix[rowOffset + j] * weight;
        }
      }
    }

    const result = new Float32Array(dim);

    // 合成された行列・ベクトル積およびバイアス加算
    for (let i = 0; i < dim; i++) {
      let sum = 0;
      const rowOffset = i * dim;
      
      for (let j = 0; j < dim; j++) {
        sum += blendedMatrix[rowOffset + j] * baseVector[j];
      }
      
      result[i] = sum + blendedBias[i];
    }

    return result;
  }

  /**
   * 複数の意図を指定された重みでブレンドし、複数のベクトルに一括で適用します。
   *
   * @param baseVectors 変換元のベクトルの配列
   * @param blendWeights 意図の名前と重みのマッピング
   * @returns 変換後のベクトルの配列
   */
  public tuneBatchBlended(baseVectors: (number[] | Float32Array)[], blendWeights: Record<string, number>): Float32Array[] {
    const dim = this.dimension;
    const blendedMatrix = new Float32Array(dim * dim);
    const blendedBias = new Float32Array(dim);

    // ブレンド行列とバイアスの合成
    for (const [intentName, weight] of Object.entries(blendWeights)) {
      const matrix = this.matrices.get(intentName);
      const bias = this.biases.get(intentName);

      if (!matrix || !bias) {
        throw new Error(`Intent '${intentName}' not found during blending.`);
      }

      for (let i = 0; i < dim; i++) {
        blendedBias[i] += bias[i] * weight;
        const rowOffset = i * dim;
        for (let j = 0; j < dim; j++) {
          blendedMatrix[rowOffset + j] += matrix[rowOffset + j] * weight;
        }
      }
    }

    const batchSize = baseVectors.length;
    const results = new Array<Float32Array>(batchSize);

    for (let k = 0; k < batchSize; k++) {
      const baseVector = baseVectors[k];
      if (baseVector.length !== dim) {
        throw new Error(
          `Vector dimension mismatch at index ${k}. Expected ${dim}, got ${baseVector.length}.`
        );
      }

      const result = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        let sum = 0;
        const rowOffset = i * dim;
        
        for (let j = 0; j < dim; j++) {
          sum += blendedMatrix[rowOffset + j] * baseVector[j];
        }
        
        result[i] = sum + blendedBias[i];
      }
      results[k] = result;
    }

    return results;
  }
}
