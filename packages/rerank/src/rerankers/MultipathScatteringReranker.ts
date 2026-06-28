import {
  allocateWasmMemory,
  withWasmMemoryStack,
  writeFloat32ArrayToWasm,
  readFloat32ArrayFromWasm,
} from "@warpvector/core";
import { BaseGraphReranker, GraphRerankerResult } from "./BaseGraphReranker";

export interface MultipathScatteringConfig {
  /**
   * 波の多重散乱の減衰率 (0.0 ~ 1.0)。
   * 値が大きいほど、遠くのドキュメント（多くのホップを経た経路）からの影響を強く受けます。
   * (PageRank や Random Walk with Restart における 1 - テレポート確率 に相当します)
   * デフォルトは 0.85
   */
  alpha?: number;

  /**
   * 候補ドキュメント間の類似度グラフを構築する際の、エッジの足切りしきい値。
   * この値未満の類似度は0とみなされ、グラフが疎（Sparse）になります。
   * デフォルトは 0.0 (すべての正の類似度を考慮)
   */
  threshold?: number;

  /**
   * 定常場を求めるための反復計算（多重経路の追跡）の最大回数。
   * デフォルトは 20
   */
  maxIterations?: number;

  /**
   * 収束判定のための許容誤差。
   * デフォルトは 1e-6
   */
  tolerance?: number;
}

export type MultipathScatteringResult = GraphRerankerResult;

/**
 * MultipathScatteringReranker
 *
 * 物理学における「多重経路散乱場理論」のアイデアにインスパイアされたリランカー。
 * ベクトル空間上の候補ドキュメント群を散乱のネットワーク（グラフ）とみなし、
 * 初期スコア（波源からの入射波）が多重反射を繰り返して定常状態に至る過程を、
 * Random Walk with Restart (PPR) を用いてシミュレーションします。
 * これにより、孤立したノイズドキュメントのスコアは減衰し、
 * 多数の類似ドキュメントから「多重経路で支持されている」真の意図（波源）のスコアが際立ちます。
 */
export class MultipathScatteringReranker extends BaseGraphReranker {
  public alpha: number;
  public maxIterations: number;
  public tolerance: number;

  constructor(config: MultipathScatteringConfig = {}) {
    super(config.threshold ?? 0.0);
    this.alpha = config.alpha ?? 0.85;
    this.maxIterations = config.maxIterations ?? 20;
    this.tolerance = config.tolerance ?? 1e-6;

    if (
      typeof this.alpha !== "number" ||
      Number.isNaN(this.alpha) ||
      this.alpha < 0 ||
      this.alpha >= 1
    ) {
      throw new Error("MultipathScatteringReranker: alpha must be in [0, 1).");
    }
    if (
      typeof this.maxIterations !== "number" ||
      Number.isNaN(this.maxIterations) ||
      this.maxIterations < 1 ||
      !Number.isInteger(this.maxIterations)
    ) {
      throw new Error(
        "MultipathScatteringReranker: maxIterations must be a positive integer.",
      );
    }
    if (
      typeof this.tolerance !== "number" ||
      Number.isNaN(this.tolerance) ||
      this.tolerance <= 0
    ) {
      throw new Error(
        "MultipathScatteringReranker: tolerance must be a positive number.",
      );
    }
  }

  protected hasRequiredWasmExports(exports: any): boolean {
    return !!(
      exports.buildMultipathTransitionMatrixWasm &&
      exports.multipathScatteringPowerIterationWasm
    );
  }

  protected prepareInitialScores(
    S0: Float32Array,
    N: number,
    initialScores?: number[],
  ): number {
    let sumS0 = 0;
    if (initialScores && initialScores.length === N) {
      let minScore = Infinity;
      for (let i = 0; i < N; i++) {
        if (initialScores[i] < minScore) minScore = initialScores[i];
      }
      for (let i = 0; i < N; i++) {
        const val = initialScores[i] - Math.min(0, minScore);
        S0[i] = val;
        sumS0 += val;
      }
    } else {
      let minScore = Infinity;
      for (let i = 0; i < N; i++) {
        if (S0[i] < minScore) minScore = S0[i];
      }
      for (let i = 0; i < N; i++) {
        S0[i] = S0[i] - Math.min(0, minScore);
        sumS0 += S0[i];
      }
    }

    if (sumS0 > 0) {
      for (let i = 0; i < N; i++) S0[i] /= sumS0;
    } else {
      const uniform = 1.0 / N;
      for (let i = 0; i < N; i++) S0[i] = uniform;
    }
    return sumS0;
  }

  protected restoreInitialScore(s0Val: number, scaleFactor: number): number {
    return s0Val * scaleFactor;
  }

  protected executeWasm(
    flatVectors: Float32Array,
    S0: Float32Array,
    N: number,
    dim: number,
    exports: any,
  ): Float32Array {
    const memory = exports.memory as WebAssembly.Memory;

    return withWasmMemoryStack(() => {
      const vectorsPtr = allocateWasmMemory(N * dim * 4);
      const pMatrixPtr = allocateWasmMemory(N * N * 4 + N * 4);
      const s0Ptr = allocateWasmMemory(N * 4);
      const currentSPtr = allocateWasmMemory(N * 4);
      const nextSPtr = allocateWasmMemory(N * 4);

      writeFloat32ArrayToWasm(memory, flatVectors, vectorsPtr);
      writeFloat32ArrayToWasm(memory, S0, s0Ptr);
      writeFloat32ArrayToWasm(memory, S0, currentSPtr);

      exports.buildMultipathTransitionMatrixWasm(
        vectorsPtr,
        N,
        dim,
        this.threshold,
        pMatrixPtr,
      );

      exports.multipathScatteringPowerIterationWasm(
        pMatrixPtr,
        s0Ptr,
        currentSPtr,
        nextSPtr,
        N,
        this.alpha,
        this.maxIterations,
        this.tolerance,
      );

      return readFloat32ArrayFromWasm(memory, currentSPtr, N);
    });
  }

  protected executeJs(
    CNorm: Float32Array[],
    S0: Float32Array,
    N: number,
  ): Float32Array {
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

    const P = new Float32Array(N * N);
    for (let j = 0; j < N; j++) {
      if (D[j] > 0) {
        for (let i = 0; i < N; i++) {
          P[i * N + j] = W[i * N + j] / D[j];
        }
      } else {
        P[j * N + j] = 1.0;
      }
    }

    const currentS = new Float32Array(S0);
    const nextS = new Float32Array(N);
    const alpha = this.alpha;
    const oneMinusAlpha = 1.0 - alpha;

    for (let iter = 0; iter < this.maxIterations; iter++) {
      let maxDiff = 0;

      for (let i = 0; i < N; i++) {
        let pSum = 0;
        const rowOffset = i * N;
        for (let j = 0; j < N; j++) {
          pSum += P[rowOffset + j] * currentS[j];
        }
        nextS[i] = alpha * pSum + oneMinusAlpha * S0[i];

        const diff = Math.abs(nextS[i] - currentS[i]);
        if (diff > maxDiff) maxDiff = diff;
      }

      currentS.set(nextS);
      if (maxDiff < this.tolerance) {
        break;
      }
    }

    return currentS;
  }
}
