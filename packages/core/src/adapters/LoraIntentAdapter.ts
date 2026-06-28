import {
  assertDimension,
  flattenMatrix,
  applyAffine,
  addScaledVector,
} from "../utils";
import {
  safeJsonParse,
  assertPositiveInt,
  assertObject,
  assertNumberArray,
} from "../validation";
import {
  WarpAdapter,
  TransformOutput,
  InputVector,
} from "../interfaces/WarpAdapter";

/**
 * 低ランク適応（LoRA）の重みを定義するインターフェース
 * @interface LoraIntentWeights
 */
export interface LoraIntentWeights {
  /**
   * 行列A: 次元数を rank から dimension に拡張する行列
   * @type {number[][] | number[] | Float32Array}
   */
  matrixA: number[][] | number[] | Float32Array; // [dim][rank]

  /**
   * 行列B: 次元数を dimension から rank に圧縮する行列
   * @type {number[][] | number[] | Float32Array}
   */
  matrixB: number[][] | number[] | Float32Array; // [rank][dim]

  /**
   * バイアスベクトル
   * @type {number[] | Float32Array}
   */
  bias: number[] | Float32Array; // [dim]
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

    let flatA: Float32Array;
    if (matrixA instanceof Float32Array) {
      flatA = new Float32Array(matrixA);
    } else if (Array.isArray(matrixA) && !Array.isArray(matrixA[0])) {
      assertDimension(matrixA as number[], this.dimension * this.rank, `Intent '${intentName}' Matrix A (1D)`);
      flatA = new Float32Array(matrixA as number[]);
    } else {
      flatA = flattenMatrix(
        matrixA as number[][],
        this.dimension,
        this.rank,
        `Intent '${intentName}' Matrix A`,
      );
    }

    let flatB: Float32Array;
    if (matrixB instanceof Float32Array) {
      flatB = new Float32Array(matrixB);
    } else if (Array.isArray(matrixB) && !Array.isArray(matrixB[0])) {
      assertDimension(matrixB as number[], this.rank * this.dimension, `Intent '${intentName}' Matrix B (1D)`);
      flatB = new Float32Array(matrixB as number[]);
    } else {
      flatB = flattenMatrix(
        matrixB as number[][],
        this.rank,
        this.dimension,
        `Intent '${intentName}' Matrix B`,
      );
    }

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
    applyAffine(matB, null, baseVector, y, this.dimension, this.rank);

    // Step 2: z = A * y + b (サイズ: dimension)
    // 複雑度: O(D * r)
    applyAffine(matA, bias, y, result, this.rank, this.dimension);

    // Step 3: 残差結合 result = x + z
    addScaledVector(result, baseVector, 1.0);

    return result;
  }

  /**
   * 現在の LoraIntentAdapter の全状態を JSON としてエクスポートします。
   */
  public exportState(): string {
    const intents: Record<
      string,
      { matrixA: number[]; matrixB: number[]; bias: number[] }
    > = {};
    for (const [name, flatA] of this.matricesA.entries()) {
      const flatB = this.matricesB.get(name)!;
      const bias = this.biases.get(name)!;
      intents[name] = {
        matrixA: Array.from(flatA), // export as flattened for simplicity during import
        matrixB: Array.from(flatB),
        bias: Array.from(bias),
      };
    }
    return JSON.stringify({
      dimension: this.dimension,
      rank: this.rank,
      intents,
    });
  }

  /**
   * エクスポートされた JSON 状態から LoraIntentAdapter を復元します。
   */
  public static importState(stateJson: string): LoraIntentAdapter {
    const data = assertObject(
      safeJsonParse(stateJson, "LoraIntentAdapter"),
      "root",
    );
    const dimension = assertPositiveInt(data.dimension, "dimension");
    const rank = assertPositiveInt(data.rank, "rank");
    const adapter = new LoraIntentAdapter(dimension, rank);

    const intents = assertObject(data.intents, "intents");
    for (const [name, rawIntent] of Object.entries(intents)) {
      const intent = assertObject(rawIntent, `intents.${name}`);
      const matrixA = assertNumberArray(
        intent.matrixA,
        `intents.${name}.matrixA`,
      );
      const matrixB = assertNumberArray(
        intent.matrixB,
        `intents.${name}.matrixB`,
      );
      const bias = assertNumberArray(intent.bias, `intents.${name}.bias`);
      adapter.addIntent(name, {
        matrixA: new Float32Array(matrixA),
        matrixB: new Float32Array(matrixB),
        bias: new Float32Array(bias),
      });
    }
    return adapter;
  }
}
