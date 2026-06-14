import {
  Activation,
  applyActivationToVector,
  softmax,
  cosineSimilarity,
} from "./utils";
import { wasmBase64 } from "./wasm/wasm-binary";

/**
 * 意図（コンテキスト）ごとの変換情報を定義するインターフェース
 * @interface IntentWeights
 */
export interface IntentWeights {
  /**
   * 空間を歪めるための変換行列 (W)
   * @type {number[][]}
   */
  matrix: number[][];

  /**
   * 空間を特定の方向へシフトさせるバイアスベクトル (b)
   * @type {number[]}
   */
  bias: number[];

  /**
   * 自己アテンション型ブレンド（自動ルーティング）用の代表ベクトル (オプション)
   * @type {number[] | undefined}
   */
  routingVector?: number[];
}

/**
 * WASMモジュールのシングルトンインスタンス
 * @type {WebAssembly.Instance | null}
 */
let wasmInstance: WebAssembly.Instance | null = null;

/**
 * WASMとJSで共有されるメモリ
 * @type {WebAssembly.Memory | null}
 */
let wasmMemory: WebAssembly.Memory | null = null;

/**
 * WASMモジュールを初期化し、インスタンスを取得するヘルパー関数
 * 初期化に失敗した場合はnullを返し、自動的に純粋なJS実行にフォールバックします
 *
 * @returns {WebAssembly.Instance | null} 初期化済みのWASMインスタンス、またはエラー時はnull
 */
function getWasmInstance(): WebAssembly.Instance | null {
  if (wasmInstance) return wasmInstance;
  try {
    // Base64エンコードされたWASMバイナリをデコード
    const bytes = Uint8Array.from(atob(wasmBase64), (c) => c.charCodeAt(0));
    const module = new WebAssembly.Module(bytes);
    // モジュールのインスタンス化 (同期実行可能なくらい小さなWASMであることを前提)
    wasmInstance = new WebAssembly.Instance(module);
    // WASM側でエクスポートされたメモリ領域への参照を取得
    wasmMemory = wasmInstance.exports.memory as WebAssembly.Memory;
    return wasmInstance;
  } catch (e) {
    console.warn(
      "Failed to initialize WASM module, falling back to JS implementation.",
      e,
    );
    return null;
  }
}

/**
 * 意図に応じたベクトル空間の動的変形（アフィン変換）を行うアダプタークラス。
 * パフォーマンスのために内部ではすべての配列を Float32Array として扱い、
 * 大規模なバッチ処理には自動的にWASM/SIMDによる最適化を利用します。
 */
export class IntentAdapter {
  private readonly dimension: number;
  private readonly matrices: Map<string, Float32Array>;
  private readonly biases: Map<string, Float32Array>;
  private readonly routingVectors: Map<string, Float32Array>;

  /**
   * IntentAdapter を初期化します。
   * 行列とバイアスを Float32Array にコンパイルし、キャッシュ局所性と計算速度を最適化します。
   *
   * @constructor
   * @param {Record<string, IntentWeights>} intents - 各インテント名と IntentWeights (行列・バイアス・ルーティングベクトル) のマッピング
   * @throws {Error} インテントが一つも提供されていない場合にエラーをスローします。
   */
  constructor(intents: Record<string, IntentWeights>) {
    this.matrices = new Map();
    this.biases = new Map();
    this.routingVectors = new Map();

    const intentKeys = Object.keys(intents);
    if (intentKeys.length === 0) {
      throw new Error("At least one intent must be provided.");
    }

    // 最初のインテントのバイアス長からベクトル空間の次元数を自動決定
    const firstIntent = intents[intentKeys[0]];
    this.dimension = firstIntent.bias.length;

    for (const [intentName, weights] of Object.entries(intents)) {
      this.addIntent(intentName, weights);
    }
  }

  /**
   * 実行時に新しい意図を動的に追加または更新します。
   *
   * @param {string} intentName - 追加する意図の名前
   * @param {IntentWeights} weights - 意図の重み（行列とバイアス、任意でルーティングベクトル）
   * @throws {Error} 行列やバイアスの次元数が現在の次元数と一致しない場合にエラーをスローします。
   * @returns {void}
   */
  public addIntent(intentName: string, weights: IntentWeights): void {
    const { matrix, bias, routingVector } = weights;

    // 次元数の整合性チェック
    if (bias.length !== this.dimension) {
      throw new Error(
        `Intent '${intentName}': Bias dimension mismatch. Expected ${this.dimension}, got ${bias.length}.`,
      );
    }

    if (matrix.length !== this.dimension) {
      throw new Error(
        `Intent '${intentName}': Matrix row dimension mismatch. Expected ${this.dimension}, got ${matrix.length}.`,
      );
    }

    // 行列をフラットなFloat32Arrayに変換（メモリの連続アクセスで高速化するため）
    const flatMatrix = new Float32Array(this.dimension * this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      if (matrix[i].length !== this.dimension) {
        throw new Error(
          `Intent '${intentName}': Matrix column dimension mismatch at row ${i}. Expected ${this.dimension}, got ${matrix[i].length}.`,
        );
      }
      for (let j = 0; j < this.dimension; j++) {
        flatMatrix[i * this.dimension + j] = matrix[i][j];
      }
    }

    this.matrices.set(intentName, flatMatrix);
    this.biases.set(intentName, new Float32Array(bias));

    // オプションのルーティングベクトルが提供されていれば保存
    if (routingVector) {
      if (routingVector.length !== this.dimension) {
        throw new Error(
          `Intent '${intentName}': Routing vector dimension mismatch. Expected ${this.dimension}, got ${routingVector.length}.`,
        );
      }
      this.routingVectors.set(intentName, new Float32Array(routingVector));
    }
  }

  /**
   * 指定した意図を削除します。
   *
   * @param {string} intentName - 削除する意図の名前
   * @returns {void}
   */
  public removeIntent(intentName: string): void {
    this.matrices.delete(intentName);
    this.biases.delete(intentName);
    this.routingVectors.delete(intentName);
  }

  /**
   * 行列とバイアスを用いてベクトルにアフィン変換を適用する内部関数 (x' = W * x + b)
   *
   * @param {Float32Array} matrix - フラット化された変換行列
   * @param {Float32Array} bias - バイアスベクトル
   * @param {number[] | Float32Array} vector - 変換元の入力ベクトル
   * @param {Float32Array} result - 計算結果を格納する配列 (出力先)
   * @returns {void}
   */
  private applyAffine(
    matrix: Float32Array,
    bias: Float32Array,
    vector: number[] | Float32Array,
    result: Float32Array,
  ): void {
    const dim = this.dimension;
    for (let i = 0; i < dim; i++) {
      let sum = 0;
      const rowOffset = i * dim;
      for (let j = 0; j < dim; j++) {
        sum += matrix[rowOffset + j] * vector[j];
      }
      result[i] = sum + bias[i];
    }
  }

  /**
   * 複数の意図を指定された重みでブレンドした一時的な行列とバイアスを計算します。
   * W_blend = Σ(w_i * W_i), b_blend = Σ(w_i * b_i)
   *
   * @param {Record<string, number>} blendWeights - ブレンドする各意図の重みマップ
   * @returns {{matrix: Float32Array, bias: Float32Array}} ブレンドされた合成行列とバイアス
   * @throws {Error} 指定された意図が見つからない場合にエラーをスローします。
   */
  private computeBlendedWeights(blendWeights: Record<string, number>): {
    matrix: Float32Array;
    bias: Float32Array;
  } {
    const dim = this.dimension;
    const blendedMatrix = new Float32Array(dim * dim);
    const blendedBias = new Float32Array(dim);

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

    return { matrix: blendedMatrix, bias: blendedBias };
  }

  /**
   * 指定された意図（intent）に基づいて、ベースベクトルにアフィン変換を適用します。
   * 数式: x' = W_I * x + b_I
   *
   * @param {number[] | Float32Array} baseVector - 変換元のベクトル
   * @param {string} intent - 適用する意図（intent）の名前
   * @param {Activation} [activation] - （オプション）変換後に適用する非線形活性化関数
   * @returns {Float32Array} 変換後のベクトル
   * @throws {Error} ベクトルの次元数が一致しない場合、または指定された意図が存在しない場合にエラーをスローします。
   */
  public tune(
    baseVector: number[] | Float32Array,
    intent: string,
    activation?: Activation,
  ): Float32Array {
    if (baseVector.length !== this.dimension) {
      throw new Error(
        `Vector dimension mismatch. Expected ${this.dimension}, got ${baseVector.length}.`,
      );
    }

    const matrix = this.matrices.get(intent);
    const bias = this.biases.get(intent);

    if (!matrix || !bias) {
      throw new Error(`Intent '${intent}' not found.`);
    }

    const result = new Float32Array(this.dimension);
    this.applyAffine(matrix, bias, baseVector, result);
    applyActivationToVector(result, activation); // 活性化関数の適用
    return result;
  }

  /**
   * 複数のベースベクトルに対して、指定された意図のアフィン変換をバッチ処理で適用します。
   * メモリと条件が許せば、自動的にWASM/SIMDエンジンにオフロードして超高速処理を行います。
   *
   * @param {(number[] | Float32Array)[]} baseVectors - 変換元のベクトルの配列 (2次元配列)
   * @param {string} intent - 適用する意図（intent）の名前
   * @param {Activation} [activation] - （オプション）非線形活性化関数
   * @returns {Float32Array[]} 変換後のベクトルの配列
   * @throws {Error} 指定された意図が存在しない場合や入力ベクトルの次元が不正な場合にエラーをスローします。
   */
  public tuneBatch(
    baseVectors: (number[] | Float32Array)[],
    intent: string,
    activation?: Activation,
  ): Float32Array[] {
    const matrix = this.matrices.get(intent);
    const bias = this.biases.get(intent);

    if (!matrix || !bias) {
      throw new Error(`Intent '${intent}' not found.`);
    }

    const batchSize = baseVectors.length;
    const instance = getWasmInstance();
    // WASMに必要なメモリ量（行列 + バイアス + 入力ベクトル全体 + 出力ベクトル全体）
    const requiredBytes =
      (this.dimension * this.dimension +
        this.dimension +
        batchSize * this.dimension * 2) *
      4;

    // WASMモジュールが利用可能で、共有メモリにアクセスできる場合
    if (instance && wasmMemory) {
      // 必要な場合はメモリサイズを拡張(1ページ = 64KB)
      if (requiredBytes > wasmMemory.buffer.byteLength) {
        const currentPages = wasmMemory.buffer.byteLength / 65536;
        const requiredPages = Math.ceil(requiredBytes / 65536);
        try {
          wasmMemory.grow(requiredPages - currentPages);
        } catch (e) {
          // メモリ拡張に失敗した場合はエラーを握りつぶして後続のJSフォールバックに任せる
        }
      }

      // WASMのメモリ容量が十分であればWASMで処理を実行
      if (requiredBytes <= wasmMemory.buffer.byteLength) {
        const f32Mem = new Float32Array(wasmMemory.buffer);
        let ptr = 0;

        // メモリ空間に変換行列をコピー
        const matrixPtr = ptr;
        f32Mem.set(matrix, ptr);
        ptr += this.dimension * this.dimension;

        // メモリ空間にバイアスをコピー
        const biasPtr = ptr;
        f32Mem.set(bias, ptr);
        ptr += this.dimension;

        // メモリ空間に入力ベクトルのバッチを連続してコピー
        const vectorsPtr = ptr;
        for (let k = 0; k < batchSize; k++) {
          f32Mem.set(baseVectors[k], ptr + k * this.dimension);
        }
        ptr += batchSize * this.dimension;

        // 出力結果を書き込むためのメモリポインタ
        const resultsPtr = ptr;

        // AssemblyScriptのコア関数を呼び出し
        const tuneBatchWasm = instance.exports
          .tuneBatchWasm as CallableFunction;
        // バイト単位のポインタ（Float32のインデックス * 4）を渡す
        tuneBatchWasm(
          matrixPtr * 4,
          biasPtr * 4,
          vectorsPtr * 4,
          resultsPtr * 4,
          this.dimension,
          batchSize,
        );

        // WASMメモリから計算結果を読み取り、必要に応じて活性化関数を適用
        const results = new Array<Float32Array>(batchSize);
        for (let k = 0; k < batchSize; k++) {
          const res = f32Mem.slice(
            resultsPtr + k * this.dimension,
            resultsPtr + (k + 1) * this.dimension,
          );
          applyActivationToVector(res, activation);
          results[k] = res;
        }
        return results;
      }
    }

    // --- WASMが使えない場合のフォールバック (純粋なJS処理) ---
    const results = new Array<Float32Array>(batchSize);
    for (let k = 0; k < batchSize; k++) {
      const baseVector = baseVectors[k];
      if (baseVector.length !== this.dimension) {
        throw new Error(
          `Vector dimension mismatch at index ${k}. Expected ${this.dimension}, got ${baseVector.length}.`,
        );
      }
      const result = new Float32Array(this.dimension);
      this.applyAffine(matrix, bias, baseVector, result);
      applyActivationToVector(result, activation);
      results[k] = result;
    }

    return results;
  }

  /**
   * 複数の意図を指定された重みでブレンドし、ベクトルにアフィン変換を適用します。
   *
   * @param {number[] | Float32Array} baseVector - 変換元のベクトル
   * @param {Record<string, number>} blendWeights - 意図の名前と重みのマッピング (例: { riskAnalysis: 0.7, economicImpact: 0.3 })
   * @param {Activation} [activation] - （オプション）非線形活性化関数
   * @returns {Float32Array} 変換後のベクトル
   * @throws {Error} 入力ベクトルの次元が不正な場合、または指定された意図が存在しない場合にエラーをスローします。
   */
  public tuneBlended(
    baseVector: number[] | Float32Array,
    blendWeights: Record<string, number>,
    activation?: Activation,
  ): Float32Array {
    if (baseVector.length !== this.dimension) {
      throw new Error(
        `Vector dimension mismatch. Expected ${this.dimension}, got ${baseVector.length}.`,
      );
    }

    // ブレンドされた合成行列とバイアスを一時的に計算
    const { matrix, bias } = this.computeBlendedWeights(blendWeights);
    const result = new Float32Array(this.dimension);

    this.applyAffine(matrix, bias, baseVector, result);
    applyActivationToVector(result, activation);
    return result;
  }

  /**
   * 複数の意図を指定された重みでブレンドし、複数のベクトルに一括で適用します。
   * これもバッチ処理と同様に自動的にWASM最適化が有効になります。
   *
   * @param {(number[] | Float32Array)[]} baseVectors - 変換元のベクトルの配列
   * @param {Record<string, number>} blendWeights - 意図の名前と重みのマッピング
   * @param {Activation} [activation] - （オプション）非線形活性化関数
   * @returns {Float32Array[]} 変換後のベクトルの配列
   * @throws {Error} ベクトルの次元数が一致しない場合、または指定された意図が存在しない場合にエラーをスローします。
   */
  public tuneBatchBlended(
    baseVectors: (number[] | Float32Array)[],
    blendWeights: Record<string, number>,
    activation?: Activation,
  ): Float32Array[] {
    const { matrix, bias } = this.computeBlendedWeights(blendWeights);
    const batchSize = baseVectors.length;

    // WASMによる最適化
    const instance = getWasmInstance();
    const requiredBytes =
      (this.dimension * this.dimension +
        this.dimension +
        batchSize * this.dimension * 2) *
      4;

    if (instance && wasmMemory) {
      if (requiredBytes > wasmMemory.buffer.byteLength) {
        const currentPages = wasmMemory.buffer.byteLength / 65536;
        const requiredPages = Math.ceil(requiredBytes / 65536);
        try {
          wasmMemory.grow(requiredPages - currentPages);
        } catch (e) {}
      }

      if (requiredBytes <= wasmMemory.buffer.byteLength) {
        const f32Mem = new Float32Array(wasmMemory.buffer);
        let ptr = 0;

        const matrixPtr = ptr;
        f32Mem.set(matrix, ptr);
        ptr += this.dimension * this.dimension;

        const biasPtr = ptr;
        f32Mem.set(bias, ptr);
        ptr += this.dimension;

        const vectorsPtr = ptr;
        for (let k = 0; k < batchSize; k++) {
          f32Mem.set(baseVectors[k], ptr + k * this.dimension);
        }
        ptr += batchSize * this.dimension;

        const resultsPtr = ptr;

        const tuneBatchWasm = instance.exports
          .tuneBatchWasm as CallableFunction;
        tuneBatchWasm(
          matrixPtr * 4,
          biasPtr * 4,
          vectorsPtr * 4,
          resultsPtr * 4,
          this.dimension,
          batchSize,
        );

        const results = new Array<Float32Array>(batchSize);
        for (let k = 0; k < batchSize; k++) {
          const res = f32Mem.slice(
            resultsPtr + k * this.dimension,
            resultsPtr + (k + 1) * this.dimension,
          );
          applyActivationToVector(res, activation);
          results[k] = res;
        }
        return results;
      }
    }

    // --- WASMフォールバック (JS) ---
    const results = new Array<Float32Array>(batchSize);
    for (let k = 0; k < batchSize; k++) {
      const baseVector = baseVectors[k];
      if (baseVector.length !== this.dimension) {
        throw new Error(
          `Vector dimension mismatch at index ${k}. Expected ${this.dimension}, got ${baseVector.length}.`,
        );
      }
      const result = new Float32Array(this.dimension);
      this.applyAffine(matrix, bias, baseVector, result);
      applyActivationToVector(result, activation);
      results[k] = result;
    }
    return results;
  }

  /**
   * 自己アテンション型動的ブレンド (Auto-blending / Routing)
   *
   * クエリとして入力されたベースベクトル自体と、各意図の `routingVector`（代表ベクトル）の
   * 類似度を比較し、Softmax関数により最適なブレンド比率を自動で算出・適用します。
   *
   * @param {number[] | Float32Array} baseVector - ユーザーからのクエリベクトルなど
   * @param {Activation} [activation] - （オプション）非線形活性化関数
   * @returns {Float32Array} 動的ブレンドによって変換されたベクトル
   * @throws {Error} ベースベクトルの次元が異なる場合、またはルーティングベクトルが一つも存在しない場合にエラーをスローします。
   */
  public tuneAutoBlended(
    baseVector: number[] | Float32Array,
    activation?: Activation,
  ): Float32Array {
    if (baseVector.length !== this.dimension) {
      throw new Error(
        `Vector dimension mismatch. Expected ${this.dimension}, got ${baseVector.length}.`,
      );
    }

    const intentNames: string[] = [];
    const scores: number[] = [];

    // 登録されている全てのルーティングベクトルとのコサイン類似度を計算（スコアリング）
    for (const [intentName, vector] of this.routingVectors.entries()) {
      intentNames.push(intentName);
      scores.push(cosineSimilarity(baseVector, vector));
    }

    if (intentNames.length === 0) {
      throw new Error("No routing vectors available for auto-blending.");
    }

    // Softmax関数を通してスコアを合計1.0の確率分布（ウェイト）に変換
    const weightsArray = softmax(scores);
    const blendWeights: Record<string, number> = {};
    for (let i = 0; i < intentNames.length; i++) {
      blendWeights[intentNames[i]] = weightsArray[i];
    }

    // 計算した動的なウェイトを用いてブレンド処理を実行
    return this.tuneBlended(baseVector, blendWeights, activation);
  }
}
