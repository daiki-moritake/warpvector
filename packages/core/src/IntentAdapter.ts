import {
  Activation,
  applyActivationToVector,
  softmax,
  cosineSimilarity,
  flattenMatrix,
  assertDimension,
  applyAffine,
  addScaledVector,
} from "./utils";
import {
  safeJsonParse,
  assertPositiveInt,
  assertObject,
  assertNumberArray,
} from "./validation";
import {
  getWasmInstance,
  ensureWasmMemory,
  writeFloat32ArrayToWasm,
  allocateWasmMemory,
  withWasmMemoryStack,
} from "./wasm/wasm-loader";
import { WarpAdapter } from "./WarpAdapter";

/**
 * 意図（コンテキスト）ごとの変換情報を定義するインターフェース
 * @interface IntentWeights
 */
export interface IntentWeights {
  /**
   * 空間を歪めるための変換行列 (W)
   * 内部処理と互換性のため、number[][] と 1次元にフラット化された Float32Array の両方をサポートします。
   * @type {number[][] | Float32Array}
   */
  matrix: number[][] | Float32Array;

  /**
   * 空間を特定の方向へシフトさせるバイアスベクトル (b)
   * @type {number[] | Float32Array}
   */
  bias: number[] | Float32Array;

  /**
   * 自己アテンション型ブレンド（自動ルーティング）用の代表ベクトル (オプション)
   * @type {number[] | Float32Array | undefined}
   */
  routingVector?: number[] | Float32Array;
}

/**
 * 意図に応じたベクトル空間の動的変形（アフィン変換）を行うアダプタークラス。
 * パフォーマンスのために内部ではすべての配列を Float32Array として扱い、
 * 大規模なバッチ処理には自動的にWASM/SIMDによる最適化を利用します。
 */
export class IntentAdapter implements WarpAdapter {
  private weightsMap: Map<string, IntentWeights> = new Map();
  private readonly dimension: number;
  private readonly matrices: Map<string, Float32Array>;
  private readonly biases: Map<string, Float32Array>;
  private readonly routingVectors: Map<string, Float32Array>;

  /**
   * IntentAdapter を初期化します。
   * 行列とバイアスを Float32Array にコンパイルし、キャッシュ局所性と計算速度を最適化します。
   *
   * @constructor
   * @param {Record<string, IntentWeights> | number} intentsOrDimension - 各インテント名と IntentWeights のマッピング、または次元数（空から始める場合）
   * @throws {Error} インテントが一つも提供されておらず、次元数も不明な場合にエラーをスローします。
   */
  constructor(intentsOrDimension: Record<string, IntentWeights> | number) {
    this.matrices = new Map();
    this.biases = new Map();
    this.routingVectors = new Map();

    if (typeof intentsOrDimension === "number") {
      this.dimension = intentsOrDimension;
    } else {
      const intentKeys = Object.keys(intentsOrDimension);
      if (intentKeys.length === 0) {
        throw new Error(
          "At least one intent or a specific dimension must be provided.",
        );
      }

      const firstIntent = intentsOrDimension[intentKeys[0]];
      this.dimension = firstIntent.bias.length;

      for (const [intentName, weights] of Object.entries(intentsOrDimension)) {
        this.addIntent(intentName, weights);
      }
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
    this.weightsMap.set(intentName, weights);

    // 次元数の整合性チェック
    assertDimension(bias, this.dimension, `Intent '${intentName}' Bias`);

    let flatMatrix: Float32Array;
    if (matrix instanceof Float32Array) {
      assertDimension(
        matrix,
        this.dimension * this.dimension,
        `Intent '${intentName}' Flat matrix`,
      );
      flatMatrix = new Float32Array(matrix);
    } else {
      flatMatrix = flattenMatrix(
        matrix,
        this.dimension,
        this.dimension,
        `Intent '${intentName}' Matrix`,
      );
    }

    this.matrices.set(intentName, flatMatrix);
    this.biases.set(intentName, new Float32Array(bias));

    // オプションのルーティングベクトルが提供されていれば保存
    if (routingVector) {
      assertDimension(
        routingVector,
        this.dimension,
        `Intent '${intentName}' Routing vector`,
      );
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
    this.weightsMap.delete(intentName);
  }

  // (Private applyAffine was removed in favor of utils.ts applyAffine)

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

      addScaledVector(blendedBias, bias, weight);
      addScaledVector(blendedMatrix, matrix, weight);
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
    assertDimension(baseVector, this.dimension, "Base vector");

    const matrix = this.matrices.get(intent);
    const bias = this.biases.get(intent);

    if (!matrix || !bias) {
      throw new Error(`Intent '${intent}' not found.`);
    }

    const result = new Float32Array(this.dimension);
    applyAffine(matrix, bias, baseVector, result, this.dimension);
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

    // WASMモジュールが利用可能で、共有メモリにアクセス・拡張できる場合
    if (instance) {
      const memory = instance.exports.memory as WebAssembly.Memory;
      return withWasmMemoryStack(() => {
        const matrixPtr = allocateWasmMemory(this.dimension * this.dimension * 4);
        const biasPtr = allocateWasmMemory(this.dimension * 4);
        const vectorsPtr = allocateWasmMemory(batchSize * this.dimension * 4);
        const resultsPtr = allocateWasmMemory(batchSize * this.dimension * 4);

        writeFloat32ArrayToWasm(memory, matrix, matrixPtr);
        writeFloat32ArrayToWasm(memory, bias, biasPtr);

        for (let k = 0; k < batchSize; k++) {
          writeFloat32ArrayToWasm(
            memory,
            baseVectors[k],
            vectorsPtr + k * this.dimension * 4,
          );
        }

        const tuneBatchWasm = instance.exports.tuneBatchWasm as CallableFunction;
        tuneBatchWasm(
          matrixPtr,
          biasPtr,
          vectorsPtr,
          resultsPtr,
          this.dimension,
          batchSize,
        );

        const results = new Array<Float32Array>(batchSize);
        const outF32Mem = new Float32Array(memory.buffer);
        for (let k = 0; k < batchSize; k++) {
          const res = outF32Mem.slice(
            resultsPtr / 4 + k * this.dimension,
            resultsPtr / 4 + (k + 1) * this.dimension,
          );
          applyActivationToVector(res, activation);
          results[k] = res;
        }
        return results;
      });
    }

    // --- WASMが使えない場合のフォールバック (純粋なJS処理) ---
    const results = new Array<Float32Array>(batchSize);
    for (let k = 0; k < batchSize; k++) {
      const baseVector = baseVectors[k];
      assertDimension(baseVector, this.dimension, `Base vector at index ${k}`);
      const result = new Float32Array(this.dimension);
      applyAffine(matrix, bias, baseVector, result, this.dimension);
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
    assertDimension(baseVector, this.dimension, "Base vector");

    // ブレンドされた合成行列とバイアスを一時的に計算
    const { matrix, bias } = this.computeBlendedWeights(blendWeights);
    const result = new Float32Array(this.dimension);

    applyAffine(matrix, bias, baseVector, result, this.dimension);
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

    if (instance) {
      const memory = instance.exports.memory as WebAssembly.Memory;
      return withWasmMemoryStack(() => {
        const matrixPtr = allocateWasmMemory(this.dimension * this.dimension * 4);
        const biasPtr = allocateWasmMemory(this.dimension * 4);
        const vectorsPtr = allocateWasmMemory(batchSize * this.dimension * 4);
        const resultsPtr = allocateWasmMemory(batchSize * this.dimension * 4);

        writeFloat32ArrayToWasm(memory, matrix, matrixPtr);
        writeFloat32ArrayToWasm(memory, bias, biasPtr);

        for (let k = 0; k < batchSize; k++) {
          writeFloat32ArrayToWasm(
            memory,
            baseVectors[k],
            vectorsPtr + k * this.dimension * 4,
          );
        }

        const tuneBatchWasm = instance.exports.tuneBatchWasm as CallableFunction;
        tuneBatchWasm(
          matrixPtr,
          biasPtr,
          vectorsPtr,
          resultsPtr,
          this.dimension,
          batchSize,
        );

        const results = new Array<Float32Array>(batchSize);
        const outF32Mem = new Float32Array(memory.buffer);
        for (let k = 0; k < batchSize; k++) {
          const res = outF32Mem.slice(
            resultsPtr / 4 + k * this.dimension,
            resultsPtr / 4 + (k + 1) * this.dimension,
          );
          applyActivationToVector(res, activation);
          results[k] = res;
        }
        return results;
      });
    }

    // --- WASMフォールバック (JS) ---
    const results = new Array<Float32Array>(batchSize);
    for (let k = 0; k < batchSize; k++) {
      const baseVector = baseVectors[k];
      assertDimension(baseVector, this.dimension, `Base vector at index ${k}`);
      const result = new Float32Array(this.dimension);
      applyAffine(matrix, bias, baseVector, result, this.dimension);
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
    assertDimension(baseVector, this.dimension, "Base vector");

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

  /**
   * 学習済みの意図（IntentWeights）を軽量なバイナリ（Uint8Array）としてシリアライズします。
   * これにより、JSONに比べてファイルサイズが劇的に小さくなり、ロードも高速になります。
   *
   * @param {string} intentName - エクスポートする意図の名前
   * @returns {Uint8Array} シリアライズされたバイナリデータ
   */
  public exportIntentBinary(intentName: string): Uint8Array {
    const matrix = this.matrices.get(intentName);
    const bias = this.biases.get(intentName);
    const routingVector = this.routingVectors.get(intentName);

    if (!matrix || !bias) {
      throw new Error(`Intent '${intentName}' not found.`);
    }

    const dim = this.dimension;
    const hasRouting = routingVector ? 1 : 0;

    // 4 bytes: dimension (Uint32)
    // 1 byte: hasRouting (Uint8)
    // dim*dim*4 bytes: matrix
    // dim*4 bytes: bias
    // [dim*4 bytes]: routingVector (optional)
    const totalBytes = 8 + dim * dim * 4 + dim * 4 + (hasRouting ? dim * 4 : 0);
    const buffer = new ArrayBuffer(totalBytes);
    const dataView = new DataView(buffer);
    const uint8View = new Uint8Array(buffer);

    dataView.setUint32(0, dim, true);
    dataView.setUint8(4, hasRouting);

    let offset = 8;

    // matrix
    uint8View.set(
      new Uint8Array(matrix.buffer, matrix.byteOffset, matrix.byteLength),
      offset,
    );
    offset += matrix.byteLength;

    // bias
    uint8View.set(
      new Uint8Array(bias.buffer, bias.byteOffset, bias.byteLength),
      offset,
    );
    offset += bias.byteLength;

    // routingVector
    if (routingVector) {
      uint8View.set(
        new Uint8Array(
          routingVector.buffer,
          routingVector.byteOffset,
          routingVector.byteLength,
        ),
        offset,
      );
    }

    return uint8View;
  }

  /**
   * バイナリデータ（Uint8Array）から IntentWeights を復元し、
   * そのままアダプターに新しい意図として追加します。
   *
   * @param {string} intentName - 追加する意図の名前
   * @param {Uint8Array} binary - exportIntentBinary で生成されたバイナリデータ
   */
  public importIntentBinary(intentName: string, binary: Uint8Array): void {
    if (binary.length < 8) {
      throw new Error("Invalid binary format: too short.");
    }
    const dataView = new DataView(
      binary.buffer,
      binary.byteOffset,
      binary.byteLength,
    );
    const dim = dataView.getUint32(0, true);
    const hasRouting = dataView.getUint8(4);

    if (this.dimension !== undefined && dim !== this.dimension) {
      throw new Error(
        `Dimension mismatch. Expected ${this.dimension}, got ${dim}.`,
      );
    }

    const expectedBytes =
      8 + dim * dim * 4 + dim * 4 + (hasRouting ? dim * 4 : 0);
    if (binary.length !== expectedBytes) {
      throw new Error(
        `Invalid binary length. Expected ${expectedBytes}, got ${binary.length}.`,
      );
    }

    let offset = 8;
    const matrix = new Float32Array(dim * dim);
    matrix.set(
      new Float32Array(binary.buffer, binary.byteOffset + offset, dim * dim),
    );
    offset += dim * dim * 4;

    const bias = new Float32Array(dim);
    bias.set(new Float32Array(binary.buffer, binary.byteOffset + offset, dim));
    offset += dim * 4;

    let routingVector: Float32Array | undefined = undefined;
    if (hasRouting) {
      routingVector = new Float32Array(dim);
      routingVector.set(
        new Float32Array(binary.buffer, binary.byteOffset + offset, dim),
      );
    }

    this.matrices.set(intentName, matrix);
    this.biases.set(intentName, bias);
    if (routingVector) {
      this.routingVectors.set(intentName, routingVector);
    }
  }

  /**
   * 現在の IntentAdapter の全状態（全インテント）を JSON としてシリアライズしてエクスポートします。
   * (WarpPipeline 等の統合管理用)
   */
  public exportState(): string {
    const intents: Record<
      string,
      { matrix: number[]; bias: number[]; routingVector?: number[] }
    > = {};
    for (const [name, matrix] of this.matrices.entries()) {
      const bias = this.biases.get(name)!;
      const routing = this.routingVectors.get(name);
      intents[name] = {
        matrix: Array.from(matrix),
        bias: Array.from(bias),
        routingVector: routing ? Array.from(routing) : undefined,
      };
    }
    return JSON.stringify({ dimension: this.dimension, intents });
  }

  /**
   * エクスポートされた JSON 状態から IntentAdapter を復元します。
   */
  public static importState(stateJson: string): IntentAdapter {
    const data = assertObject(
      safeJsonParse(stateJson, "IntentAdapter"),
      "root",
    );
    const dimension = assertPositiveInt(data.dimension, "dimension");
    const adapter = new IntentAdapter(dimension);

    const intents = assertObject(data.intents, "intents");

    for (const [name, rawIntent] of Object.entries(intents)) {
      const intent = assertObject(rawIntent, `intents.${name}`);
      const matrix = assertNumberArray(intent.matrix, `intents.${name}.matrix`);
      const bias = assertNumberArray(intent.bias, `intents.${name}.bias`);

      adapter.addIntent(name, {
        matrix: new Float32Array(matrix),
        bias: new Float32Array(bias),
        routingVector: intent.routingVector
          ? new Float32Array(
              assertNumberArray(
                intent.routingVector,
                `intents.${name}.routingVector`,
              ),
            )
          : undefined,
      });
    }
    return adapter;
  }
}
