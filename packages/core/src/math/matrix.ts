import { WarpDimensionMismatchError } from "../errors";

/**
 * 2次元配列（行列）を指定した次元数で1次元のFloat32Arrayにフラット化します。
 *
 * @param {number[][]} matrix - 2次元配列の行列
 * @param {number} rows - 期待される行数
 * @param {number} cols - 期待される列数
 * @param {string} contextName - エラーメッセージ用のコンテキスト名
 * @returns {Float32Array} フラット化された1次元配列
 * @throws {WarpDimensionMismatchError} 次元数が一致しない場合にエラーをスローします。
 */
export function flattenMatrix(
  matrix: number[][],
  rows: number,
  cols: number,
  contextName: string = "Matrix",
): Float32Array {
  if (matrix.length !== rows) {
    throw new WarpDimensionMismatchError(
      `${contextName} (rows)`,
      rows,
      matrix.length,
    );
  }
  const flat = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    if (matrix[i].length !== cols) {
      throw new WarpDimensionMismatchError(
        `${contextName} (row ${i} cols)`,
        cols,
        matrix[i].length,
      );
    }
    for (let j = 0; j < cols; j++) {
      flat[i * cols + j] = matrix[i][j];
    }
  }
  return flat;
}

/**
 * 変換行列とバイアスを持つオブジェクトから、処理しやすいFloat32Arrayのセットを生成します。
 *
 * @param weights 行列(2Dまたは1D)とバイアスを含むオブジェクト
 * @param dim 次元数
 * @param contextName エラー時のコンテキスト名
 * @returns 1次元化されたFloat32Arrayの行列とバイアス
 */
export function getFlatMatrixAndBias(
  weights: { matrix: number[][] | Float32Array; bias: number[] | Float32Array },
  dim: number,
  contextName: string = "Weights",
): { flatMatrix: Float32Array; bias: Float32Array } {
  let flatMatrix: Float32Array;
  if (weights.matrix instanceof Float32Array) {
    flatMatrix = new Float32Array(weights.matrix);
  } else {
    flatMatrix = flattenMatrix(weights.matrix, dim, dim, contextName);
  }
  const bias = new Float32Array(weights.bias);
  return { flatMatrix, bias };
}

/**
 * ベクトルに対してアフィン変換 (x' = W * x + b) を適用します。
 * 次元削減/拡張 (M x N) にも対応しています。
 *
 * @param matrix 1次元にフラット化された変換行列 (outDim x inDim)
 * @param bias バイアスベクトル (outDim)。省略可能
 * @param vector 変換元の入力ベクトル (inDim)
 * @param result 計算結果を格納する配列 (出力先, outDim)
 * @param inDim 入力ベクトルの次元数
 * @param outDim 出力ベクトルの次元数 (省略時は inDim と同じ)
 */
export function applyAffine(
  matrix: Float32Array,
  bias: Float32Array | undefined | null,
  vector: number[] | Float32Array,
  result: Float32Array,
  inDim: number,
  outDim: number = inDim,
): void {
  if (bias) {
    for (let i = 0; i < outDim; i++) {
      let sum = bias[i];
      const rowOffset = i * inDim;
      for (let j = 0; j < inDim; j++) {
        sum += matrix[rowOffset + j] * vector[j];
      }
      result[i] = sum;
    }
  } else {
    for (let i = 0; i < outDim; i++) {
      let sum = 0;
      const rowOffset = i * inDim;
      for (let j = 0; j < inDim; j++) {
        sum += matrix[rowOffset + j] * vector[j];
      }
      result[i] = sum;
    }
  }
}
