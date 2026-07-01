import type { ProjectionWeights } from "@warpvector/core";
import { svd } from "../math/svd";

/**
 * Procrustes ペア（旧モデルのベクトルと新モデルのベクトル）。
 */
export interface ProcrustesPair {
  /** 移行元のベクトル */
  source: number[] | Float32Array;
  /** 移行先のベクトル */
  target: number[] | Float32Array;
}

/**
 * ProcrustesAligner のオプション。
 */
export interface ProcrustesAlignerOptions {
  /**
   * 直交制約を適用するかどうか。
   * true の場合、SVD ベースの直交 Procrustes 解析を使用し、
   * 結果の変換行列は直交行列（回転+鏡映）になります。
   * これにより空間の幾何的構造（距離・角度）が保存されます。
   * @default true
   */
  orthogonal?: boolean;

  /**
   * バイアス項を含めるかどうか。
   * true の場合、ペア間の平均を差し引いた後に Procrustes を適用し、
   * 平均差をバイアスとして保存します。
   * @default true
   */
  includeBias?: boolean;
}

/**
 * 直交 Procrustes 解析による空間アライメントを行うクラス。
 *
 * 閉形式解 (SVD ベース) により、反復学習なしで最適な直交変換行列を計算します。
 * MigrationTrainer の初期値として使用するか、単体で空間翻訳に使用します。
 *
 * 数学的根拠:
 * - 目的: min ‖Y - XW‖_F  subject to W^T W = I
 * - 解: M = X^T Y の SVD を計算 → M = U Σ V^T → W* = U V^T
 *
 * @example
 * ```typescript
 * import { ProcrustesAligner } from "warpvector/train";
 *
 * const aligner = new ProcrustesAligner(1536, 1536);
 * aligner.addPair({ source: oldVec1, target: newVec1 });
 * aligner.addPair({ source: oldVec2, target: newVec2 });
 * // ... 100ペア程度追加
 *
 * // 閉形式解で一発計算（反復不要）
 * const weights = aligner.solve();
 *
 * // AlignmentAdapter にそのまま投入
 * adapter.addAlignment("v1_to_v2", weights);
 * ```
 */
export class ProcrustesAligner {
  private readonly sourceDim: number;
  private readonly targetDim: number;
  private pairs: ProcrustesPair[] = [];

  /**
   * @param sourceDimension 移行元の次元数
   * @param targetDimension 移行先の次元数
   */
  constructor(sourceDimension: number, targetDimension: number) {
    if (sourceDimension <= 0 || !Number.isInteger(sourceDimension)) {
      throw new Error(
        `sourceDimension must be a positive integer, got ${sourceDimension}.`,
      );
    }
    if (targetDimension <= 0 || !Number.isInteger(targetDimension)) {
      throw new Error(
        `targetDimension must be a positive integer, got ${targetDimension}.`,
      );
    }
    this.sourceDim = sourceDimension;
    this.targetDim = targetDimension;
  }

  /**
   * ペアを追加します。
   *
   * @param pair 移行元と移行先のベクトルペア
   */
  public addPair(pair: ProcrustesPair): void {
    if (pair.source.length !== this.sourceDim) {
      throw new Error(
        `Source dimension mismatch: expected ${this.sourceDim}, got ${pair.source.length}.`,
      );
    }
    if (pair.target.length !== this.targetDim) {
      throw new Error(
        `Target dimension mismatch: expected ${this.targetDim}, got ${pair.target.length}.`,
      );
    }
    this.pairs.push(pair);
  }

  /**
   * 複数のペアを一括追加します。
   */
  public addPairs(pairs: ProcrustesPair[]): void {
    for (const pair of pairs) {
      this.addPair(pair);
    }
  }

  /**
   * 登録済みペア数を返します。
   */
  public get pairCount(): number {
    return this.pairs.length;
  }

  /**
   * Procrustes 解析を実行し、最適な変換行列を返します。
   *
   * @param options オプション
   * @returns ProjectionWeights（matrix と bias）
   * @throws ペアが不足している場合
   */
  public solve(
    options: ProcrustesAlignerOptions = {},
  ): ProjectionWeights {
    if (this.pairs.length < 2) {
      throw new Error(
        `At least 2 pairs are required for Procrustes alignment. ` +
          `Currently ${this.pairs.length} pair(s).`,
      );
    }

    const sDim = this.sourceDim;
    const tDim = this.targetDim;
    const n = this.pairs.length;
    const includeBias = options.includeBias ?? true;

    // Step 1: 平均を計算（バイアスに使用）
    const meanSource = new Float32Array(sDim);
    const meanTarget = new Float32Array(tDim);

    if (includeBias) {
      for (const pair of this.pairs) {
        for (let i = 0; i < sDim; i++) meanSource[i] += pair.source[i];
        for (let i = 0; i < tDim; i++) meanTarget[i] += pair.target[i];
      }
      for (let i = 0; i < sDim; i++) meanSource[i] /= n;
      for (let i = 0; i < tDim; i++) meanTarget[i] /= n;
    }

    // Step 2: 中心化された行列 X (n x sDim) と Y (n x tDim) を構築
    // M = X^T Y を計算 (sDim x tDim)
    const M = new Float32Array(sDim * tDim);

    for (let p = 0; p < n; p++) {
      const src = this.pairs[p].source;
      const tgt = this.pairs[p].target;
      for (let i = 0; i < sDim; i++) {
        const xi = src[i] - (includeBias ? meanSource[i] : 0);
        const rowOffset = i * tDim;
        for (let j = 0; j < tDim; j++) {
          const yj = tgt[j] - (includeBias ? meanTarget[j] : 0);
          M[rowOffset + j] += xi * yj;
        }
      }
    }

    // Step 3: SVD of M → M = U Σ V^T
    const numComponents = Math.min(sDim, tDim);
    const svdResult = svd(M, sDim, tDim, numComponents);

    // Step 4: W = U V^T (直交 Procrustes 解)
    // W は sDim x tDim の行列
    const W = new Float32Array(sDim * tDim);

    for (let i = 0; i < sDim; i++) {
      for (let j = 0; j < tDim; j++) {
        let sum = 0;
        for (let k = 0; k < svdResult.k; k++) {
          // U[i, k] * Vt[k, j]
          sum += svdResult.U[i * svdResult.k + k] * svdResult.Vt[k * tDim + j];
        }
        W[i * tDim + j] = sum;
      }
    }

    // Step 5: バイアス = meanTarget - W^T * meanSource
    // ただし ProjectionWeights の matrix は tDim x sDim（出力行 × 入力列）の形式
    // WarpVector の applyAffine は out[i] = Σ matrix[i * sDim + j] * input[j] + bias[i]
    // つまり matrix は tDim x sDim で、W を転置する必要がある
    const matrix = new Float32Array(tDim * sDim);
    for (let i = 0; i < tDim; i++) {
      for (let j = 0; j < sDim; j++) {
        matrix[i * sDim + j] = W[j * tDim + i];
      }
    }

    const result: ProjectionWeights = { matrix };

    if (includeBias) {
      const bias = new Float32Array(tDim);
      for (let i = 0; i < tDim; i++) {
        let dot = 0;
        for (let j = 0; j < sDim; j++) {
          dot += matrix[i * sDim + j] * meanSource[j];
        }
        bias[i] = meanTarget[i] - dot;
      }
      result.bias = bias;
    }

    return result;
  }
}
