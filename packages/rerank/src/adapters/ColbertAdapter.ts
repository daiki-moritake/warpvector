import {
  getWasmInstance,
  writeFloat32ArrayToWasm,
  allocateWasmMemory,
  withWasmMemoryStack,
} from "@warpvector/core";

export class ColbertAdapter {
  private wasm: WebAssembly.Instance | null;

  constructor() {
    this.wasm = getWasmInstance();
  }

  private getWasmExports(): {
    memory: WebAssembly.Memory;
    colbertMaxSimWasm: CallableFunction;
  } | null {
    if (!this.wasm) {
      return null;
    }
    return this.wasm.exports as {
      memory: WebAssembly.Memory;
      colbertMaxSimWasm: CallableFunction;
    };
  }

  private scoreFallback(
    queryTokens: Float32Array,
    documentTokens: Float32Array,
    dim: number,
  ): number {
    const numQueryTokens = queryTokens.length / dim;
    const numDocTokens = documentTokens.length / dim;
    let totalScore = 0;

    for (let q = 0; q < numQueryTokens; q++) {
      let maxSim = -Infinity;
      for (let d = 0; d < numDocTokens; d++) {
        let sim = 0;
        for (let i = 0; i < dim; i++) {
          sim += queryTokens[q * dim + i] * documentTokens[d * dim + i];
        }
        if (sim > maxSim) maxSim = sim;
      }
      totalScore += maxSim;
    }
    return totalScore;
  }

  /**
   * 単一のクエリと単一のドキュメント間の Late Interaction (MaxSim) スコアを計算します。
   *
   * @param queryTokens クエリのトークンベクトル行列 (要素数 = queryLength * dim の平坦化された配列)
   * @param documentTokens ドキュメントのトークンベクトル行列
   * @param dim ベクトルの次元数
   * @returns MaxSimスコア
   */
  public score(
    queryTokens: Float32Array,
    documentTokens: Float32Array,
    dim: number,
  ): number {
    const numQueryTokens = queryTokens.length / dim;
    const numDocTokens = documentTokens.length / dim;

    if (numQueryTokens % 1 !== 0) {
      throw new Error(
        `Invalid queryTokens length. Must be a multiple of dim (${dim})`,
      );
    }
    if (numDocTokens % 1 !== 0) {
      throw new Error(
        `Invalid documentTokens length. Must be a multiple of dim (${dim})`,
      );
    }

    const exports = this.getWasmExports();
    if (!exports) {
      return this.scoreFallback(queryTokens, documentTokens, dim);
    }
    const { memory, colbertMaxSimWasm } = exports;

    const queryBytes = queryTokens.byteLength;
    const docBytes = documentTokens.byteLength;

    return withWasmMemoryStack(() => {
      const queryPtr = allocateWasmMemory(queryBytes);
      const docPtr = allocateWasmMemory(docBytes);

      writeFloat32ArrayToWasm(memory, queryTokens, queryPtr);
      writeFloat32ArrayToWasm(memory, documentTokens, docPtr);

      return colbertMaxSimWasm(
        queryPtr,
        docPtr,
        numQueryTokens,
        numDocTokens,
        dim,
      );
    });
  }

  /**
   * クエリと複数のドキュメント間の MaxSim スコアを計算し、スコアの降順にソートして返します。
   *
   * @param queryTokens クエリのトークンベクトル行列
   * @param documentTokensArray ドキュメントのトークンベクトル行列の配列
   * @param dim ベクトルの次元数
   * @returns スコアの降順にソートされたドキュメントのインデックスとスコアの配列
   */
  public rank(
    queryTokens: Float32Array,
    documentTokensArray: Float32Array[],
    dim: number,
  ): { index: number; score: number }[] {
    const numQueryTokens = queryTokens.length / dim;
    if (numQueryTokens % 1 !== 0) {
      throw new Error(
        `Invalid queryTokens length. Must be a multiple of dim (${dim})`,
      );
    }

    const exports = this.getWasmExports();
    if (!exports) {
      // JS Fallback
      return documentTokensArray
        .map((doc, index) => ({
          index,
          score: this.scoreFallback(queryTokens, doc, dim),
        }))
        .sort((a, b) => b.score - a.score);
    }
    const { memory, colbertMaxSimWasm } = exports;

    // ドキュメントの中で最大の長さを探す
    let maxDocLen = 0;
    for (const doc of documentTokensArray) {
      if (doc.length > maxDocLen) maxDocLen = doc.length;
    }

    const queryBytes = queryTokens.byteLength;
    const maxDocBytes = maxDocLen * Float32Array.BYTES_PER_ELEMENT;

    return withWasmMemoryStack(() => {
      const queryPtr = allocateWasmMemory(queryBytes);
      const docPtr = allocateWasmMemory(maxDocBytes);

      // クエリは一度だけ書き込む
      writeFloat32ArrayToWasm(memory, queryTokens, queryPtr);

      const f32 = new Float32Array(memory.buffer);
      const docFloatOffset = docPtr / 4;

      const results = documentTokensArray.map((doc, index) => {
        const numDocTokens = doc.length / dim;
        if (numDocTokens % 1 !== 0) {
          throw new Error(`Invalid documentTokens length at index ${index}`);
        }

        // ドキュメントをメモリにコピー
        f32.set(doc, docFloatOffset);

        // MaxSimを計算 (WASM)
        const score = colbertMaxSimWasm(
          queryPtr,
          docPtr,
          numQueryTokens,
          numDocTokens,
          dim,
        );

        return { index, score };
      });

      // スコアの降順にソート
      return results.sort((a, b) => b.score - a.score);
    });
  }
}
