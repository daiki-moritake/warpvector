import {
  getWasmInstance,
  writeFloat32ArrayToWasm,
  allocateWasmMemory,
  withWasmMemoryStack,
} from "@warpvector/core";

export class ColbertAdapter {
  constructor() {}

  private getWasmExports(): {
    memory: WebAssembly.Memory;
    colbertMaxSimWasm: CallableFunction;
    colbertMaxSimBatchWasm?: CallableFunction;
  } | null {
    const wasm = getWasmInstance();
    if (!wasm) {
      return null;
    }
    return wasm.exports as {
      memory: WebAssembly.Memory;
      colbertMaxSimWasm: CallableFunction;
      colbertMaxSimBatchWasm?: CallableFunction;
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
    const { memory, colbertMaxSimWasm, colbertMaxSimBatchWasm } = exports;

    // Use colbertMaxSimBatchWasm if available (for extreme performance)
    if (typeof colbertMaxSimBatchWasm === "function") {
      const numDocs = documentTokensArray.length;
      if (numDocs === 0) return [];

      const docTokensLengths = new Int32Array(numDocs);
      let totalDocTokens = 0;
      for (let i = 0; i < numDocs; i++) {
        const docLen = documentTokensArray[i].length;
        const tokens = docLen / dim;
        if (tokens % 1 !== 0) {
          throw new Error(`Invalid documentTokens length at index ${i}`);
        }
        docTokensLengths[i] = tokens;
        totalDocTokens += tokens;
      }

      const queryBytes = queryTokens.byteLength;
      const docsBytes = totalDocTokens * dim * Float32Array.BYTES_PER_ELEMENT;
      const docTokensMetaBytes = numDocs * Int32Array.BYTES_PER_ELEMENT;
      const resultsBytes = numDocs * Float32Array.BYTES_PER_ELEMENT;

      return withWasmMemoryStack(() => {
        const queryPtr = allocateWasmMemory(queryBytes);
        const docsPtr = allocateWasmMemory(docsBytes);
        const docTokensPtr = allocateWasmMemory(docTokensMetaBytes);
        const resultsPtr = allocateWasmMemory(resultsBytes);

        writeFloat32ArrayToWasm(memory, queryTokens, queryPtr);

        const memI32 = new Int32Array(memory.buffer);
        memI32.set(docTokensLengths, docTokensPtr / 4);

        const memF32 = new Float32Array(memory.buffer);
        let offset = docsPtr / 4;
        for (let i = 0; i < numDocs; i++) {
          memF32.set(documentTokensArray[i], offset);
          offset += documentTokensArray[i].length;
        }

        colbertMaxSimBatchWasm(
          queryPtr,
          docsPtr,
          docTokensPtr,
          resultsPtr,
          numDocs,
          numQueryTokens,
          dim
        );

        // Fetch results with a fresh view to avoid detached buffer issues
        const resF32 = new Float32Array(memory.buffer, resultsPtr, numDocs);
        const results = new Array(numDocs);
        for (let i = 0; i < numDocs; i++) {
          results[i] = { index: i, score: resF32[i] };
        }

        return results.sort((a, b) => b.score - a.score);
      });
    }

    // Fallback to iterative WASM calls
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

      const results = documentTokensArray.map((doc, index) => {
        const numDocTokens = doc.length / dim;
        if (numDocTokens % 1 !== 0) {
          throw new Error(`Invalid documentTokens length at index ${index}`);
        }

        // ドキュメントをメモリにコピー
        writeFloat32ArrayToWasm(memory, doc, docPtr);

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
