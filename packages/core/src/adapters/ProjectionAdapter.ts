import { assertDimension, flattenMatrix, applyAffine } from "../utils";
import {
  safeJsonParse,
  assertPositiveInt,
  assertObject,
  assertNumberArray,
} from "../validation";
import {
  getWasmInstance,
  ensureWasmMemory,
  writeFloat32ArrayToWasm,
  allocateWasmMemory,
  withWasmMemoryStack,
  readFloat32ArrayFromWasm,
} from "../wasm/wasm-loader";
import { WarpAdapter } from "../interfaces/WarpAdapter";

/**
 * 次元削減/拡張のための射影行列の重みを定義するインターフェース
 * @interface ProjectionWeights
 */
export interface ProjectionWeights {
  /**
   * 射影変換行列
   * 行数が outDimension、列数が inDimension となります。
   * 内部処理と互換性のため、number[][] と 1次元にフラット化された Float32Array の両方をサポートします。
   * @type {number[][] | Float32Array}
   */
  matrix: number[][] | Float32Array; // [outDimension][inDimension] or flat

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
export class ProjectionAdapter implements WarpAdapter {
  private readonly inDimension: number;
  private readonly outDimension: number;
  private wasmInstance: WebAssembly.Instance | null = null;

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

    let flatMatrix: Float32Array;
    if (matrix instanceof Float32Array) {
      assertDimension(
        matrix,
        this.outDimension * this.inDimension,
        `Projection '${name}' Matrix`,
      );
      flatMatrix = matrix;
    } else {
      flatMatrix = flattenMatrix(
        matrix as number[][],
        this.outDimension,
        this.inDimension,
        `Projection '${name}' Matrix`,
      );
    }
    this.matrices.set(name, flatMatrix);

    if (weights.bias) {
      assertDimension(
        weights.bias,
        this.outDimension,
        `Projection '${name}' Bias`,
      );
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
   * ベクトルの次元削減（射影）を実行します。
   * (WarpAdapter の実装として project の代わりに tune を提供します)
   *
   * @param vector 変換前のベクトル (例: 1536次元)
   * @param version 適用する変換バージョンの識別子 (オプション)
   * @returns 変換後のベクトル (例: 512次元)
   */
  public tune(
    vector: number[] | Float32Array,
    version: string = "default",
  ): Float32Array {
    assertDimension(vector, this.inDimension, "Base vector");

    const matrix = this.matrices.get(version);
    const bias = this.biases.get(version);
    if (!matrix) {
      throw new Error(`Projection '${version}' not found.`);
    }

    const instance = getWasmInstance();
    const matrixSize = this.inDimension * this.outDimension * 4;
    const biasSize = bias ? this.outDimension * 4 : 0;
    const vectorSize = this.inDimension * 4;
    const outputSize = this.outDimension * 4;

    const requiredBytes = matrixSize + biasSize + vectorSize + outputSize;

    if (
      instance &&
      instance.exports.projectWasm &&
      ensureWasmMemory(requiredBytes)
    ) {
      const memory = instance.exports.memory as WebAssembly.Memory;

      return withWasmMemoryStack(() => {
        const matrixPtr = allocateWasmMemory(matrixSize);
        const biasPtr = bias ? allocateWasmMemory(biasSize) : 0;
        const inputPtr = allocateWasmMemory(vectorSize);
        const outputPtr = allocateWasmMemory(outputSize);

        writeFloat32ArrayToWasm(memory, matrix, matrixPtr);
        if (bias) writeFloat32ArrayToWasm(memory, bias, biasPtr);
        writeFloat32ArrayToWasm(memory, vector, inputPtr);

        const projectWasm = instance.exports.projectWasm as CallableFunction;
        projectWasm(
          matrixPtr,
          biasPtr,
          inputPtr,
          outputPtr,
          this.inDimension,
          this.outDimension,
        );

        return readFloat32ArrayFromWasm(memory, outputPtr, this.outDimension);
      });
    }

    // --- WASMが使えない場合のフォールバック (純粋なJS処理) ---
    const result = new Float32Array(this.outDimension);

    // 行列ベクトル積: O(M * N)
    applyAffine(
      matrix,
      bias,
      vector,
      result,
      this.inDimension,
      this.outDimension,
    );

    return result;
  }

  /**
   * 現在の射影行列の状態をシリアライズしてエクスポートします。
   */
  public exportState(): string {
    const projections: Record<string, { matrix: number[]; bias?: number[] }> =
      {};
    for (const [name, matrix] of this.matrices.entries()) {
      const bias = this.biases.get(name);
      projections[name] = {
        matrix: Array.from(matrix),
        bias: bias ? Array.from(bias) : undefined,
      };
    }
    return JSON.stringify({
      inDimension: this.inDimension,
      outDimension: this.outDimension,
      projections,
    });
  }

  /**
   * エクスポートされた状態から ProjectionAdapter を復元します。
   * 注意: 保存されている matrix は既にフラット化された 1D 配列であることを前提としています。
   */
  public static importState(stateJson: string): ProjectionAdapter {
    const data = assertObject(
      safeJsonParse(stateJson, "ProjectionAdapter"),
      "root",
    );
    const inDimension = assertPositiveInt(data.inDimension, "inDimension");
    const outDimension = assertPositiveInt(data.outDimension, "outDimension");
    const adapter = new ProjectionAdapter(inDimension, outDimension);

    const projections = assertObject(data.projections, "projections");
    for (const [name, rawProj] of Object.entries(projections)) {
      const proj = assertObject(rawProj, `projections.${name}`);
      const matrix = assertNumberArray(proj.matrix, `projections.${name}.matrix`);
      adapter.matrices.set(name, new Float32Array(matrix));
      if (proj.bias) {
        const bias = assertNumberArray(proj.bias, `projections.${name}.bias`);
        adapter.biases.set(name, new Float32Array(bias));
      }
    }
    return adapter;
  }
}
