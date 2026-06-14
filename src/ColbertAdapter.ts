import {
  getWasmInstance,
  ensureWasmMemory,
  writeFloat32ArrayToWasm,
} from "./wasm/wasm-loader";

export class ColbertAdapter {
  private wasm: any;

  constructor() {
    this.wasm = getWasmInstance();
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

    const { memory, colbertMaxSimWasm } = this.wasm.exports;

    const queryBytes = queryTokens.byteLength;
    const docBytes = documentTokens.byteLength;
    const totalBytes = queryBytes + docBytes;

    ensureWasmMemory(totalBytes);

    const queryPtr = 0;
    const docPtr = queryBytes;

    writeFloat32ArrayToWasm(memory, queryTokens, queryPtr);
    writeFloat32ArrayToWasm(memory, documentTokens, docPtr);

    return colbertMaxSimWasm(
      queryPtr,
      docPtr,
      numQueryTokens,
      numDocTokens,
      dim,
    );
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

    const { memory, colbertMaxSimWasm } = this.wasm.exports;

    // ドキュメントの中で最大の長さを探す
    let maxDocLen = 0;
    for (const doc of documentTokensArray) {
      if (doc.length > maxDocLen) maxDocLen = doc.length;
    }

    const queryBytes = queryTokens.byteLength;
    const maxDocBytes = maxDocLen * Float32Array.BYTES_PER_ELEMENT;
    const totalBytes = queryBytes + maxDocBytes;

    // WASMのメモリが足りない場合は拡張
    ensureWasmMemory(totalBytes);

    const queryPtr = 0;
    const docPtr = queryBytes;

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
  }
}
