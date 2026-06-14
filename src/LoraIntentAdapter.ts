

export interface LoraIntentWeights {
  matrixA: number[][]; // [dim][rank]
  matrixB: number[][]; // [rank][dim]
  bias: number[];      // [dim]
}

export class LoraIntentAdapter {
  private readonly dimension: number;
  private readonly rank: number;
  
  // フラット化されたAとBの行列、およびバイアスを保存
  private readonly matricesA: Map<string, Float32Array>;
  private readonly matricesB: Map<string, Float32Array>;
  private readonly biases: Map<string, Float32Array>;

  /**
   * LoraIntentAdapter を初期化します。
   * 非常に高い次元（例：1536次元など）の埋め込みベクトルに対して、
   * フルマトリックスの代わりに低ランク行列（A, B）を使用することで
   * メモリ使用量と計算量を劇的に削減します。
   *
   * @param dimension ベクトル空間の次元数（D）
   * @param rank 低ランク適応のランク数（r）
   * @param intents （オプション）初期化時に追加するインテント
   */
  constructor(dimension: number, rank: number, intents?: Record<string, LoraIntentWeights>) {
    this.dimension = dimension;
    this.rank = rank;
    this.matricesA = new Map();
    this.matricesB = new Map();
    this.biases = new Map();

    if (intents) {
      for (const [intentName, weights] of Object.entries(intents)) {
        this.addIntent(intentName, weights);
      }
    }
  }

  public addIntent(intentName: string, weights: LoraIntentWeights): void {
    const { matrixA, matrixB, bias } = weights;

    if (bias.length !== this.dimension) {
      throw new Error(`Intent '${intentName}': Bias dimension mismatch. Expected ${this.dimension}, got ${bias.length}.`);
    }
    if (matrixA.length !== this.dimension || matrixA[0].length !== this.rank) {
      throw new Error(`Intent '${intentName}': Matrix A must be of size ${this.dimension}x${this.rank}`);
    }
    if (matrixB.length !== this.rank || matrixB[0].length !== this.dimension) {
      throw new Error(`Intent '${intentName}': Matrix B must be of size ${this.rank}x${this.dimension}`);
    }

    const flatA = new Float32Array(this.dimension * this.rank);
    for (let i = 0; i < this.dimension; i++) {
      for (let j = 0; j < this.rank; j++) {
        flatA[i * this.rank + j] = matrixA[i][j];
      }
    }

    const flatB = new Float32Array(this.rank * this.dimension);
    for (let i = 0; i < this.rank; i++) {
      for (let j = 0; j < this.dimension; j++) {
        flatB[i * this.dimension + j] = matrixB[i][j];
      }
    }

    this.matricesA.set(intentName, flatA);
    this.matricesB.set(intentName, flatB);
    this.biases.set(intentName, new Float32Array(bias));
  }

  public removeIntent(intentName: string): void {
    this.matricesA.delete(intentName);
    this.matricesB.delete(intentName);
    this.biases.delete(intentName);
  }

  /**
   * LoRAアプローチを用いてベクトルにアフィン変換を適用します。
   * x' = x + A(Bx) + b
   * 
   * @param baseVector 変換元のベクトル
   * @param intent 適用する意図
   * @param gateType （オプション）非線形コンテキストゲーティングを適用する場合に指定
   */
  public tune(baseVector: number[] | Float32Array, intent: string): Float32Array {
    if (baseVector.length !== this.dimension) {
      throw new Error(`Vector dimension mismatch. Expected ${this.dimension}, got ${baseVector.length}.`);
    }

    const matA = this.matricesA.get(intent);
    const matB = this.matricesB.get(intent);
    const bias = this.biases.get(intent);

    if (!matA || !matB || !bias) {
      throw new Error(`Intent '${intent}' not found.`);
    }

    const result = new Float32Array(this.dimension);
    
    // Step 1: y = B * x (サイズ: rank)
    // 複雑度: O(r * D)
    const y = new Float32Array(this.rank);
    for (let i = 0; i < this.rank; i++) {
      let sum = 0;
      const rowOffset = i * this.dimension;
      for (let j = 0; j < this.dimension; j++) {
        sum += matB[rowOffset + j] * baseVector[j];
      }
      y[i] = sum;
    }

    // Step 2: z = A * y (サイズ: dimension)
    // 複雑度: O(D * r)
    // そして残差結合 result = x + z + b を一度に行う
    for (let i = 0; i < this.dimension; i++) {
      let sum = 0;
      const rowOffset = i * this.rank;
      for (let j = 0; j < this.rank; j++) {
        sum += matA[rowOffset + j] * y[j];
      }
      result[i] = baseVector[i] + sum + bias[i];
    }

    return result;
  }
}
