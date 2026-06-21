import { innerProduct, normalize, getWasmInstance, allocateWasmMemory, withWasmMemoryStack, writeFloat32ArrayToWasm, readFloat32ArrayFromWasm } from "@warpvector/core";

export interface TimeReversalConfig {
  /**
   * 時間反転の強さ（逆拡散パラメータ）。
   * 値が大きいほど、意味の波源（ソース）へのスコア集中が強くなります。
   * デフォルトは 1.0
   */
  tau?: number;

  /**
   * 候補ドキュメント間の類似度グラフを構築する際の、エッジの足切りしきい値。
   * この値未満の類似度は0とみなされ、グラフが疎（Sparse）になります。
   * デフォルトは 0.0 (すべての正の類似度を考慮)
   */
  threshold?: number;

  /**
   * グラフの次数による正規化を行うかどうか。
   * true の場合、ハブとなるドキュメントのスコアが発散するのを防ぐランダムウォーク・ラプラシアンを使用します。
   * デフォルトは true
   */
  normalizeGraph?: boolean;

  /**
   * 時間反転（逆拡散）を反復適用する回数。
   * オーバーシュートを防ぐために、小さなtauで複数回反復する使い方が有効です。
   * デフォルトは 1
   */
  iterations?: number;
}

export interface RerankerResult {
  /** 元の候補配列におけるインデックス */
  originalIndex: number;
  /** 時間反転によって更新された新しいスコア */
  score: number;
  /** 初期のコサイン類似度スコア */
  initialScore: number;
  /** 候補ベクトル */
  vector: Float32Array;
}

/**
 * TimeReversalReranker
 * 
 * 複数の候補ドキュメント間の相互類似度ネットワーク（グラフ）を媒質とし、
 * クエリから各ドキュメントへの初期類似度を「観測された散乱波」とみなして、
 * グラフラプラシアンによる時間反転（逆拡散）を適用することで真のドキュメントに焦点を結ばせるリランカー。
 */
export class TimeReversalReranker {
  public tau: number;
  public threshold: number;
  public normalizeGraph: boolean;
  public iterations: number;
  private wasm: WebAssembly.Instance | null;

  constructor(config: TimeReversalConfig = {}) {
    this.tau = config.tau ?? 1.0;
    this.threshold = config.threshold ?? 0.0;
    this.normalizeGraph = config.normalizeGraph ?? true;
    this.iterations = config.iterations ?? 1;
    this.wasm = getWasmInstance();

    if (this.tau < 0) {
      throw new Error("TimeReversalReranker: tau must be non-negative.");
    }
    if (this.iterations < 1) {
      throw new Error("TimeReversalReranker: iterations must be at least 1.");
    }
  }

  /**
   * 候補ベクトル群に対してクエリベクトルを用い、時間反転波によるリランキングを行います。
   * ベクトルDB等ですでにスコアが計算済みの場合は initialScores を渡すことでクエリベクトルを省略（null）可能です。
   * 
   * @param query クエリベクトル（initialScoresがある場合はnull可）
   * @param candidates 検索システム等から返された候補ベクトル群
   * @param initialScores （任意）計算済みの初期コサイン類似度スコア
   * @returns スコア降順でソートされたRerankerResultの配列
   */
  public rerank(
    query: Float32Array | number[] | null,
    candidates: (Float32Array | number[])[],
    initialScores?: number[]
  ): RerankerResult[] {
    const N = candidates.length;
    if (N === 0) return [];

    if (!query && (!initialScores || initialScores.length !== N)) {
      throw new Error("TimeReversalReranker: Must provide either 'query' or a valid 'initialScores' array of the same length as candidates.");
    }

    // L2正規化を保証
    const CNorm = candidates.map(c => normalize(new Float32Array(c)));

    // 1. 初期波面の観測 (Initial Scattered Field)
    let S0 = new Float32Array(N);
    if (initialScores && initialScores.length === N) {
      S0.set(initialScores);
    } else if (query) {
      const qNorm = normalize(new Float32Array(query));
      for (let i = 0; i < N; i++) {
        S0[i] = innerProduct(qNorm, CNorm[i]);
      }
    }

    if (N === 1) {
      return [{ originalIndex: 0, score: S0[0], initialScore: S0[0], vector: CNorm[0] }];
    }

    if (this.wasm) {
      const exports = this.wasm.exports as any;
      if (exports.buildTimeReversalGraphWasm && exports.timeReversalIterationWasm) {
        const dim = CNorm[0].length;
        const flatVectors = new Float32Array(N * dim);
        for (let i = 0; i < N; i++) {
          flatVectors.set(CNorm[i], i * dim);
        }
        return this.rerankWasm(flatVectors, CNorm, S0, N, dim);
      }
    }

    return this.rerankJs(CNorm, S0, N);
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
    initialScores?: number[]
  ): RerankerResult[] {
    const N = numDocs;
    if (N === 0) return [];
    if (flatCandidates.length !== N * dim) {
      throw new Error("TimeReversalReranker: flatCandidates length mismatch.");
    }

    if (!query && (!initialScores || initialScores.length !== N)) {
      throw new Error("TimeReversalReranker: Must provide either 'query' or a valid 'initialScores' array.");
    }

    let S0 = new Float32Array(N);
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
      return [{ originalIndex: 0, score: S0[0], initialScore: S0[0], vector: flatCandidates.slice(0, dim) }];
    }

    if (this.wasm) {
      const exports = this.wasm.exports as any;
      if (exports.buildTimeReversalGraphWasm && exports.timeReversalIterationWasm) {
        return this.rerankWasm(flatCandidates, null, S0, N, dim);
      }
    }

    const CNorm: Float32Array[] = [];
    for (let i = 0; i < N; i++) {
      CNorm.push(flatCandidates.slice(i * dim, (i + 1) * dim));
    }
    return this.rerankJs(CNorm, S0, N);
  }

  private rerankWasm(
    flatVectors: Float32Array,
    CNorm: Float32Array[] | null,
    S0: Float32Array,
    N: number,
    dim: number
  ): RerankerResult[] {
    const exports = this.wasm!.exports as any;
    const memory = exports.memory as WebAssembly.Memory;

    return withWasmMemoryStack(() => {
      const vectorsPtr = allocateWasmMemory(N * dim * 4);
      const wMatrixPtr = allocateWasmMemory(N * N * 4);
      const dArrayPtr = allocateWasmMemory(N * 4);
      const currentSPtr = allocateWasmMemory(N * 4);
      const nextSPtr = allocateWasmMemory(N * 4);

      writeFloat32ArrayToWasm(memory, flatVectors, vectorsPtr);
      writeFloat32ArrayToWasm(memory, S0, currentSPtr);

      exports.buildTimeReversalGraphWasm(
        vectorsPtr,
        N,
        dim,
        this.threshold,
        wMatrixPtr,
        dArrayPtr
      );

      exports.timeReversalIterationWasm(
        wMatrixPtr,
        dArrayPtr,
        currentSPtr,
        nextSPtr,
        N,
        this.tau,
        this.iterations,
        this.normalizeGraph
      );

      const currentS = readFloat32ArrayFromWasm(memory, currentSPtr, N);

      const results: RerankerResult[] = [];
      for (let i = 0; i < N; i++) {
        results.push({
          originalIndex: i,
          score: currentS[i],
          initialScore: S0[i],
          vector: CNorm ? CNorm[i] : flatVectors.slice(i * dim, (i + 1) * dim)
        });
      }

      results.sort((a, b) => b.score - a.score);
      return results;
    });
  }

  private rerankJs(CNorm: Float32Array[], S0: Float32Array, N: number): RerankerResult[] {
    // 2. ドキュメント・マニフォールドの構築（グラフ媒質）
    // W[i][j] = max(0, cos_sim(c_i, c_j) - threshold)
    // 最適化: 1次元配列で管理し、関数呼び出しを減らしたインライン計算
    const W = new Float32Array(N * N);
    const D = new Float32Array(N); // 次数 (Degree)
    const dim = CNorm[0].length;
    const threshold = this.threshold;

    for (let i = 0; i < N; i++) {
      const ci = CNorm[i];
      for (let j = i + 1; j < N; j++) {
        const cj = CNorm[j];
        let sim = 0;
        // インライン内積計算 (パフォーマンス重視)
        for (let k = 0; k < dim; k++) {
          sim += ci[k] * cj[k];
        }

        const w = Math.max(0, sim - threshold);
        if (w > 0) {
          W[i * N + j] = w;
          W[j * N + i] = w; // 対称行列
          D[i] += w;
          D[j] += w;
        }
      }
    }

    // 3. グラフラプラシアンによる時間反転・逆拡散の反復
    let currentS = new Float32Array(S0);
    const nextS = new Float32Array(N);
    const tau = this.tau;

    for (let iter = 0; iter < this.iterations; iter++) {
      for (let i = 0; i < N; i++) {
        let diffSum = 0;
        const rowOffset = i * N;
        
        for (let j = 0; j < N; j++) {
          if (i === j) continue;
          const w = W[rowOffset + j];
          if (w > 0) {
            diffSum += w * (currentS[i] - currentS[j]);
          }
        }
        
        if (this.normalizeGraph && D[i] > 0) {
          diffSum /= D[i]; // ランダムウォーク・ラプラシアン (ハブネス対策)
        }
        nextS[i] = Math.max(0, currentS[i] + tau * diffSum);
      }
      
      // スワップ
      currentS.set(nextS);
    }

    // 4. 結果のフォーマットとソート
    const results: RerankerResult[] = [];
    for (let i = 0; i < N; i++) {
      results.push({
        originalIndex: i,
        score: currentS[i],
        initialScore: S0[i],
        vector: CNorm[i]
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results;
  }
}
