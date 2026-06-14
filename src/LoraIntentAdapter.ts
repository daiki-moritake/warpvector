import { assertDimension, flattenMatrix } from "./utils";
import { WarpAdapter } from "./WarpAdapter";

/**
 * 低ランク適応（LoRA）の重みを定義するインターフェース
 * @interface LoraIntentWeights
 */
export interface LoraIntentWeights {
  /**
   * 行列A: 次元数を rank から dimension に拡張する行列
   * @type {number[][]}
   */
  matrixA: number[][]; // [dim][rank]

  /**
   * 行列B: 次元数を dimension から rank に圧縮する行列
   * @type {number[][]}
   */
  matrixB: number[][]; // [rank][dim]

  /**
   * バイアスベクトル
   * @type {number[]}
   */
  bias: number[]; // [dim]
}

/**
 * LoraIntentAdapter クラス
 * 低ランク行列（A, B）を使用して高次元ベクトルのアフィン変換をメモリ効率良く行います。
 */
export class LoraIntentAdapter implements WarpAdapter {
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
   * @constructor
   * @param {number} dimension - ベクトル空間の元の次元数（D）
   * @param {number} rank - 低ランク適応における中間ランク数（r）
   * @param {Record<string, LoraIntentWeights>} [intents] - 初期化時に追加するインテントのマップ
   */
  constructor(
    dimension: number,
    rank: number,
    intents?: Record<string, LoraIntentWeights>,
  ) {
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

  /**
   * 実行時に新しいLoRA意図を動的に追加または更新します。
   *
   * @param {string} intentName - 追加または更新する意図の名前
   * @param {LoraIntentWeights} weights - LoRA意図の重み（行列A、行列B、バイアス）
   * @throws {Error} 行列またはバイアスの次元数が一致しない場合にエラーをスローします。
   * @returns {void}
   */
  public addIntent(intentName: string, weights: LoraIntentWeights): void {
    const { matrixA, matrixB, bias } = weights;

    assertDimension(bias, this.dimension, `Intent '${intentName}' Bias`);

    const flatA = flattenMatrix(
      matrixA,
      this.dimension,
      this.rank,
      `Intent '${intentName}' Matrix A`,
    );
    const flatB = flattenMatrix(
      matrixB,
      this.rank,
      this.dimension,
      `Intent '${intentName}' Matrix B`,
    );

    this.matricesA.set(intentName, flatA);
    this.matricesB.set(intentName, flatB);
    this.biases.set(intentName, new Float32Array(bias));
  }

  /**
   * 指定したLoRA意図を削除します。
   *
   * @param {string} intentName - 削除する意図の名前
   * @returns {void}
   */
  public removeIntent(intentName: string): void {
    this.matricesA.delete(intentName);
    this.matricesB.delete(intentName);
    this.biases.delete(intentName);
  }

  /**
   * LoRAアプローチを用いてベクトルにアフィン変換を適用します。
   * 数式: x' = x + A(Bx) + b
   *
   * @param {number[] | Float32Array} baseVector - 変換元のベクトル
   * @param {string} intent - 適用する意図の名前
   * @returns {Float32Array} LoRA変換と残差結合が適用された新しいベクトル
   * @throws {Error} ベクトルの次元数が一致しない、または指定された意図が存在しない場合にエラーをスローします。
   */
  public tune(
    baseVector: number[] | Float32Array,
    intent: string,
  ): Float32Array {
    assertDimension(baseVector, this.dimension, "Base vector");

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

  /**
   * 現在の LoraIntentAdapter の全状態を JSON としてエクスポートします。
   */
  public exportState(): string {
    const intents: Record<string, { matrixA: number[], matrixB: number[], bias: number[] }> = {};
    for (const [name, flatA] of this.matricesA.entries()) {
      const flatB = this.matricesB.get(name)!;
      const bias = this.biases.get(name)!;
      intents[name] = {
        matrixA: Array.from(flatA), // export as flattened for simplicity during import
        matrixB: Array.from(flatB),
        bias: Array.from(bias)
      };
    }
    return JSON.stringify({ dimension: this.dimension, rank: this.rank, intents });
  }

  /**
   * エクスポートされた JSON 状態から LoraIntentAdapter を復元します。
   */
  public static importState(stateJson: string): LoraIntentAdapter {
    const data = JSON.parse(stateJson);
    const adapter = new LoraIntentAdapter(data.dimension, data.rank);
    for (const [name, intent] of Object.entries(data.intents) as any) {
      adapter.matricesA.set(name, new Float32Array(intent.matrixA));
      adapter.matricesB.set(name, new Float32Array(intent.matrixB));
      adapter.biases.set(name, new Float32Array(intent.bias));
    }
    return adapter;
  }
}
