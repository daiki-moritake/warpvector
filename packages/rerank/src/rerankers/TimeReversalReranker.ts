import { allocateWasmMemory, withWasmMemoryStack, writeFloat32ArrayToWasm, readFloat32ArrayFromWasm } from "@warpvector/core";
import { BaseGraphReranker, GraphRerankerResult } from "./BaseGraphReranker";

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

export type RerankerResult = GraphRerankerResult;

/**
 * TimeReversalReranker
 * 
 * 複数の候補ドキュメント間の相互類似度ネットワーク（グラフ）を媒質とし、
 * クエリから各ドキュメントへの初期類似度を「観測された散乱波」とみなして、
 * グラフラプラシアンによる時間反転（逆拡散）を適用することで真のドキュメントに焦点を結ばせるリランカー。
 */
export class TimeReversalReranker extends BaseGraphReranker {
  public tau: number;
  public normalizeGraph: boolean;
  public iterations: number;

  constructor(config: TimeReversalConfig = {}) {
    super(config.threshold ?? 0.0);
    this.tau = config.tau ?? 1.0;
    this.normalizeGraph = config.normalizeGraph ?? true;
    this.iterations = config.iterations ?? 1;

    if (this.tau < 0) {
      throw new Error("TimeReversalReranker: tau must be non-negative.");
    }
    if (this.iterations < 1) {
      throw new Error("TimeReversalReranker: iterations must be at least 1.");
    }
  }

  protected hasRequiredWasmExports(exports: any): boolean {
    return !!(exports.buildTimeReversalGraphWasm && exports.timeReversalIterationWasm);
  }

  protected prepareInitialScores(S0: Float32Array, N: number, initialScores?: number[]): number {
    // TimeReversalReranker では初期スコアの正規化を行わない
    return 1.0;
  }

  protected restoreInitialScore(s0Val: number, scaleFactor: number): number {
    return s0Val;
  }

  protected executeWasm(flatVectors: Float32Array, S0: Float32Array, N: number, dim: number, exports: any): Float32Array {
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

      return readFloat32ArrayFromWasm(memory, currentSPtr, N);
    });
  }

  protected executeJs(CNorm: Float32Array[], S0: Float32Array, N: number): Float32Array {
    const W = new Float32Array(N * N);
    const D = new Float32Array(N);
    const dim = CNorm[0].length;
    const threshold = this.threshold;

    for (let i = 0; i < N; i++) {
      const ci = CNorm[i];
      for (let j = i + 1; j < N; j++) {
        const cj = CNorm[j];
        let sim = 0;
        for (let k = 0; k < dim; k++) {
          sim += ci[k] * cj[k];
        }

        const w = Math.max(0, sim - threshold);
        if (w > 0) {
          W[i * N + j] = w;
          W[j * N + i] = w;
          D[i] += w;
          D[j] += w;
        }
      }
    }

    const currentS = new Float32Array(S0);
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
          diffSum /= D[i];
        }
        nextS[i] = Math.max(0, currentS[i] + tau * diffSum);
      }
      
      currentS.set(nextS);
    }

    return currentS;
  }
}
