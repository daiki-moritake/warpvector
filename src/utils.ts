/**
 * 渡されたベクトルのL2ノルム（ユークリッド距離）を計算し、
 * 長さが1になるように正規化（normalize）した新しい Float32Array を返します。
 * コサイン類似度の計算前などに使用します。
 *
 * @param vector 正規化するベクトル (number[] または Float32Array)
 * @returns 正規化された Float32Array
 */
export function normalize(vector: number[] | Float32Array): Float32Array {
  const dim = vector.length;
  let sumOfSquares = 0;

  for (let i = 0; i < dim; i++) {
    const val = vector[i];
    sumOfSquares += val * val;
  }

  const norm = Math.sqrt(sumOfSquares);
  const result = new Float32Array(dim);

  if (norm === 0) {
    // ゼロベクトルの場合はゼロベクトルを返す
    return result;
  }

  for (let i = 0; i < dim; i++) {
    result[i] = vector[i] / norm;
  }

  return result;
}
