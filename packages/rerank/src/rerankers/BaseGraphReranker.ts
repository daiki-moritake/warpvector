import { innerProduct, normalize, getWasmInstance } from "@warpvector/core";

export interface GraphRerankerResult {
  /** 元の候補配列におけるインデックス */
  originalIndex: number;
  /** グラフアルゴリズムによって更新された新しいスコア */
  score: number;
  /** 初期のコサイン類似度スコア */
  initialScore: number;
  /** 候補ベクトル */
  vector: Float32Array;
}

export abstract class BaseGraphReranker {
  public threshold: number;
  protected wasm: WebAssembly.Instance | null;

  constructor(threshold: number = 0.0) {
    this.threshold = threshold;
    this.wasm = getWasmInstance();
  }

  /**
   * 候補ベクトル群に対してクエリベクトルを用い、グラフベースのアルゴリズムでリランキングを行います。
   * ベクトルDB等ですでにスコアが計算済みの場合は initialScores を渡すことでクエリベクトルを省略（null）可能です。
   *
   * @param query クエリベクトル（initialScoresがある場合はnull可）
   * @param candidates 検索システム等から返された候補ベクトル群
   * @param initialScores （任意）計算済みの初期コサイン類似度スコア
   * @returns スコア降順でソートされた GraphRerankerResult の配列
   */
  public rerank(
    query: Float32Array | number[] | null,
    candidates: (Float32Array | number[])[],
    initialScores?: number[],
  ): GraphRerankerResult[] {
    const N = candidates.length;
    if (N === 0) return [];

    if (!query && (!initialScores || initialScores.length !== N)) {
      throw new Error(
        "BaseGraphReranker: Must provide either 'query' or a valid 'initialScores' array.",
      );
    }

    const CNorm = candidates.map((c) => normalize(new Float32Array(c)));
    const S0 = new Float32Array(N);

    if (initialScores && initialScores.length === N) {
      S0.set(initialScores);
    } else if (query) {
      const qNorm = normalize(new Float32Array(query));
      for (let i = 0; i < N; i++) {
        S0[i] = innerProduct(qNorm, CNorm[i]);
      }
    }

    if (N === 1) {
      return [
        {
          originalIndex: 0,
          score: S0[0],
          initialScore: S0[0],
          vector: CNorm[0],
        },
      ];
    }

    const scaleFactor = this.prepareInitialScores(S0, N, initialScores);

    if (this.wasm) {
      const exports = this.wasm.exports as any;
      if (this.hasRequiredWasmExports(exports)) {
        const dim = CNorm[0].length;
        const flatVectors = new Float32Array(N * dim);
        for (let i = 0; i < N; i++) {
          flatVectors.set(CNorm[i], i * dim);
        }

        const currentS = this.executeWasm(flatVectors, S0, N, dim, exports);
        return this.formatResults(
          currentS,
          S0,
          CNorm,
          null,
          N,
          dim,
          initialScores,
          scaleFactor,
        );
      }
    }

    const currentS = this.executeJs(CNorm, S0, N);
    return this.formatResults(
      currentS,
      S0,
      CNorm,
      null,
      N,
      CNorm[0].length,
      initialScores,
      scaleFactor,
    );
  }

  /**
   * ベクトルが既に平坦化された1次元 Float32Array の状態で提供される場合の、
   * ゼロコピー最適化版リランク関数。
   * エッジ環境やベクトルDBから直接バッファを受け取れる場合に極めて高速に動作します。
   * ※入力される flatCandidates は L2 正規化済みである必要があります。
   *
   * @param query クエリベクトル
   * @param flatCandidates 平坦化された候補ベクトル群 (サイズ: numDocs * dim)
   * @param numDocs ドキュメント数
   * @param dim 次元数
   * @param initialScores （任意）初期スコア
   */
  public rerankFlat(
    query: Float32Array | number[] | null,
    flatCandidates: Float32Array,
    numDocs: number,
    dim: number,
    initialScores?: number[],
  ): GraphRerankerResult[] {
    const N = numDocs;
    if (N === 0) return [];
    if (flatCandidates.length !== N * dim) {
      throw new Error("BaseGraphReranker: flatCandidates length mismatch.");
    }

    if (!query && (!initialScores || initialScores.length !== N)) {
      throw new Error(
        "BaseGraphReranker: Must provide either 'query' or a valid 'initialScores' array.",
      );
    }

    const S0 = new Float32Array(N);
    if (initialScores && initialScores.length === N) {
      S0.set(initialScores);
    } else if (query) {
      const qNorm = normalize(new Float32Array(query));
      for (let i = 0; i < N; i++) {
        const cNorm = flatCandidates.subarray(i * dim, (i + 1) * dim);
        S0[i] = innerProduct(qNorm, cNorm);
      }
    }

    if (N === 1) {
      return [
        {
          originalIndex: 0,
          score: S0[0],
          initialScore: S0[0],
          vector: flatCandidates.slice(0, dim),
        },
      ];
    }

    const scaleFactor = this.prepareInitialScores(S0, N, initialScores);

    if (this.wasm) {
      const exports = this.wasm.exports as any;
      if (this.hasRequiredWasmExports(exports)) {
        const currentS = this.executeWasm(flatCandidates, S0, N, dim, exports);
        return this.formatResults(
          currentS,
          S0,
          null,
          flatCandidates,
          N,
          dim,
          initialScores,
          scaleFactor,
        );
      }
    }

    const CNorm: Float32Array[] = [];
    for (let i = 0; i < N; i++) {
      CNorm.push(flatCandidates.slice(i * dim, (i + 1) * dim));
    }
    const currentS = this.executeJs(CNorm, S0, N);
    return this.formatResults(
      currentS,
      S0,
      CNorm,
      null,
      N,
      dim,
      initialScores,
      scaleFactor,
    );
  }

  private formatResults(
    currentS: Float32Array,
    S0: Float32Array,
    CNorm: Float32Array[] | null,
    flatCandidates: Float32Array | null,
    N: number,
    dim: number,
    initialScores: number[] | undefined,
    scaleFactor: number,
  ): GraphRerankerResult[] {
    const results: GraphRerankerResult[] = [];
    for (let i = 0; i < N; i++) {
      results.push({
        originalIndex: i,
        score: currentS[i],
        initialScore:
          initialScores && initialScores.length === N
            ? initialScores[i]
            : this.restoreInitialScore(S0[i], scaleFactor),
        vector: CNorm
          ? CNorm[i]
          : flatCandidates!.slice(i * dim, (i + 1) * dim),
      });
    }
    results.sort((a, b) => b.score - a.score);
    return results;
  }

  // 子クラスで実装すべきメソッド群

  protected abstract hasRequiredWasmExports(exports: any): boolean;

  /**
   * S0 (内積または初期スコア) に対して独自の加工(非負化や正規化)を行う。
   * @returns scaleFactor formatResults で復元する際の手掛かりとして使う値
   */
  protected abstract prepareInitialScores(
    S0: Float32Array,
    N: number,
    initialScores?: number[],
  ): number;

  /**
   * formatResults 時に、S0[i] から元の initialScore を復元するためのロジック。
   */
  protected abstract restoreInitialScore(
    s0Val: number,
    scaleFactor: number,
  ): number;

  /**
   * WASMのメモリ管理(allocateWasmMemory等)と計算の呼び出しを行い、最終的なスコア配列(currentS)を返す。
   */
  protected abstract executeWasm(
    flatVectors: Float32Array,
    S0: Float32Array,
    N: number,
    dim: number,
    exports: any,
  ): Float32Array;

  /**
   * JSでの計算コア。最終的なスコア配列(currentS)を返す。
   */
  protected abstract executeJs(
    CNorm: Float32Array[],
    S0: Float32Array,
    N: number,
  ): Float32Array;
}
