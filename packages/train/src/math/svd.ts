/**
 * 特異値分解 (SVD) のための軽量な TypeScript 実装。
 *
 * Golub-Kahan bidiagonalization + QR iteration によるSVDを実装しています。
 * 高次元（1536D）の行列に対しても使用可能ですが、PyTorch/NumPyに比べて
 * 低速であるため、学習後の事後検証や閉形式解（Procrustes）の計算など、
 * クリティカルパスではない処理に限定して使用してください。
 *
 * @module
 */

/**
 * SVD の結果。
 * A = U * diag(S) * V^T を表します。
 */
export interface SvdResult {
  /** 左特異ベクトル行列 U (m x k, column-major style stored as flat row-major) */
  U: Float32Array;
  /** 特異値ベクトル S (k 個) */
  S: Float32Array;
  /** 右特異ベクトル行列 V^T (k x n) */
  Vt: Float32Array;
  /** U の行数 */
  m: number;
  /** V^T の列数 */
  n: number;
  /** 特異値の数（min(m, n)） */
  k: number;
}

/**
 * 行列 A (m x n) の特異値分解を計算します。
 *
 * Power iteration + Gram-Schmidt 正規直交化による truncated SVD 実装。
 * 全特異値ではなく上位 k 個のみを計算するため、
 * 大きな行列でも必要な特異値の数を制限することで計算量を抑えられます。
 *
 * @param A 入力行列 (m x n, row-major, flat Float32Array)
 * @param m 行数
 * @param n 列数
 * @param numComponents 計算する特異値の数 (デフォルト: min(m, n))
 * @param maxIter Power iteration の最大反復回数
 * @returns SVD結果 { U, S, Vt }
 */
export function svd(
  A: Float32Array,
  m: number,
  n: number,
  numComponents?: number,
  maxIter: number = 100,
): SvdResult {
  const k = Math.min(numComponents ?? Math.min(m, n), Math.min(m, n));

  const U = new Float32Array(m * k);
  const S = new Float32Array(k);
  const Vt = new Float32Array(k * n);

  // 残差行列（デフレーション用）のコピー
  const R = new Float32Array(A);

  for (let comp = 0; comp < k; comp++) {
    // Power iteration で最大特異値・特異ベクトルを求める
    // eslint-disable-next-line prefer-const
    let v: Float32Array<ArrayBufferLike> = new Float32Array(n);
    // 初期化: 擬似ランダム
    for (let j = 0; j < n; j++) {
      v[j] = ((j * 7 + comp * 13 + 3) % 101) / 50.0 - 1.0;
    }
    normalizeVec(v);

    let sigma = 0;

    for (let iter = 0; iter < maxIter; iter++) {
      // u = R * v
      const u = matVecMul(R, m, n, v);
      sigma = vecNorm(u);
      if (sigma < 1e-10) break;
      scaleVec(u, 1.0 / sigma);

      // v = R^T * u
      const vNew = matTransposeVecMul(R, m, n, u);
      const sigmaNew = vecNorm(vNew);
      if (sigmaNew < 1e-10) break;
      scaleVec(vNew, 1.0 / sigmaNew);

      // 収束判定
      let diff = 0;
      for (let j = 0; j < n; j++) {
        diff += (vNew[j] - v[j]) ** 2;
      }
      v = vNew;

      if (diff < 1e-12) break;
    }

    // u を再計算（最終版）
    const u = matVecMul(R, m, n, v);
    sigma = vecNorm(u);
    if (sigma > 1e-10) {
      scaleVec(u, 1.0 / sigma);
    }

    // 結果を格納
    S[comp] = sigma;
    for (let i = 0; i < m; i++) U[i * k + comp] = u[i];
    for (let j = 0; j < n; j++) Vt[comp * n + j] = v[j];

    // デフレーション: R = R - sigma * u * v^T
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        R[i * n + j] -= sigma * u[i] * v[j];
      }
    }
  }

  return { U, S, Vt, m, n, k };
}

/**
 * 条件数 (Condition Number) を計算します。
 * σ_max / σ_min で定義されます。
 *
 * @param singularValues 特異値配列
 * @returns 条件数（σ_min が 0 に近い場合は Infinity）
 */
export function conditionNumber(singularValues: Float32Array): number {
  if (singularValues.length === 0) return Infinity;

  let max = singularValues[0];
  let min = singularValues[0];
  for (let i = 1; i < singularValues.length; i++) {
    if (singularValues[i] > max) max = singularValues[i];
    if (singularValues[i] < min) min = singularValues[i];
  }

  if (min < 1e-10) return Infinity;
  return max / min;
}

/**
 * スペクトルノルム (Spectral Norm) を返します。
 * 最大特異値 σ_max です。
 *
 * @param singularValues 特異値配列
 * @returns スペクトルノルム
 */
export function spectralNorm(singularValues: Float32Array): number {
  if (singularValues.length === 0) return 0;
  let max = singularValues[0];
  for (let i = 1; i < singularValues.length; i++) {
    if (singularValues[i] > max) max = singularValues[i];
  }
  return max;
}

/**
 * 行列の単位行列からの乖離度を計算します。
 * ‖W - I‖_F / √d で定義されます。
 *
 * @param matrix 正方行列（d x d, row-major flat）
 * @param d 次元数
 * @returns フロベニウスノルムベースの乖離度
 */
export function identityDeviation(
  matrix: Float32Array,
  d: number,
): number {
  let sumSq = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      const expected = i === j ? 1.0 : 0.0;
      const diff = matrix[i * d + j] - expected;
      sumSq += diff * diff;
    }
  }
  return Math.sqrt(sumSq) / Math.sqrt(d);
}

// ---- 内部ヘルパー ----

function matVecMul(
  A: Float32Array,
  m: number,
  n: number,
  v: Float32Array,
): Float32Array {
  const result = new Float32Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    const rowOffset = i * n;
    for (let j = 0; j < n; j++) {
      sum += A[rowOffset + j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

function matTransposeVecMul(
  A: Float32Array,
  m: number,
  n: number,
  u: Float32Array,
): Float32Array {
  const result = new Float32Array(n);
  for (let i = 0; i < m; i++) {
    const rowOffset = i * n;
    const ui = u[i];
    for (let j = 0; j < n; j++) {
      result[j] += A[rowOffset + j] * ui;
    }
  }
  return result;
}

function vecNorm(v: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}

function normalizeVec(v: Float32Array): void {
  const norm = vecNorm(v);
  if (norm > 1e-10) {
    scaleVec(v, 1.0 / norm);
  }
}

function scaleVec(v: Float32Array, scale: number): void {
  for (let i = 0; i < v.length; i++) {
    v[i] *= scale;
  }
}
