/**
 * 渡されたベクトルのL2ノルム（ユークリッド距離）を計算し、
 * 長さが1になるように正規化（normalize）した新しい Float32Array を返します。
 * コサイン類似度の計算前などに使用します。
 *
 * @param {number[] | Float32Array} vector - 正規化するベクトル
 * @returns {Float32Array} 正規化された新しい Float32Array ベクトル。ゼロベクトルの場合はゼロベクトルのまま返します。
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

/**
 * 球面線形補間 (Slerp)
 * 高次元の埋め込み空間において、幾何学的な構造（コサイン類似度）を維持したまま
 * 2つのベクトル間を滑らかに補間します。
 *
 * @param {number[] | Float32Array} v1 - 始点ベクトル
 * @param {number[] | Float32Array} v2 - 終点ベクトル
 * @param {number} t - 補間係数 (0.0 〜 1.0)
 * @returns {Float32Array} 補間後の新しいベクトル
 * @throws {Error} ベクトルの次元数が異なる場合、またはゼロベクトルが指定された場合にエラーをスローします。
 */
export function slerp(
  v1: number[] | Float32Array,
  v2: number[] | Float32Array,
  t: number,
): Float32Array {
  const dim = v1.length;
  if (dim !== v2.length) {
    throw new Error("Vectors must have the same dimension for slerp.");
  }

  let dot = 0;
  let norm1Sq = 0;
  let norm2Sq = 0;
  for (let i = 0; i < dim; i++) {
    dot += v1[i] * v2[i];
    norm1Sq += v1[i] * v1[i];
    norm2Sq += v2[i] * v2[i];
  }

  const norm1 = Math.sqrt(norm1Sq);
  const norm2 = Math.sqrt(norm2Sq);

  if (norm1 === 0 || norm2 === 0) {
    throw new Error("Cannot apply slerp on zero vectors.");
  }

  let cosTheta = dot / (norm1 * norm2);
  cosTheta = Math.max(-1.0, Math.min(1.0, cosTheta)); // 丸め誤差によるNaN防止

  const theta = Math.acos(cosTheta);
  const sinTheta = Math.sin(theta);

  const result = new Float32Array(dim);

  // 角度が非常に小さい場合、ゼロ除算を防ぐためLerp（線形補間）にフォールバック
  if (sinTheta < 1e-6) {
    for (let i = 0; i < dim; i++) {
      result[i] = v1[i] + t * (v2[i] - v1[i]);
    }
    return result;
  }

  const w1 = Math.sin((1 - t) * theta) / sinTheta;
  const w2 = Math.sin(t * theta) / sinTheta;
  const interpolatedMagnitude = norm1 + t * (norm2 - norm1);

  for (let i = 0; i < dim; i++) {
    const scaledV1 = v1[i] / norm1;
    const scaledV2 = v2[i] / norm2;
    result[i] = (scaledV1 * w1 + scaledV2 * w2) * interpolatedMagnitude;
  }

  return result;
}

/**
 * 内積 (Inner Product)
 * 2つのベクトルの内積を計算します。
 *
 * @param {number[] | Float32Array} v1 - 1つ目のベクトル
 * @param {number[] | Float32Array} v2 - 2つ目のベクトル
 * @returns {number} 2つのベクトルの内積
 * @throws {Error} ベクトルの次元数が異なる場合にエラーをスローします。
 */
export function innerProduct(
  v1: number[] | Float32Array,
  v2: number[] | Float32Array,
): number {
  const dim = v1.length;
  if (dim !== v2.length) {
    throw new Error("Vectors must have the same dimension.");
  }
  let dot = 0;
  for (let i = 0; i < dim; i++) {
    dot += v1[i] * v2[i];
  }
  return dot;
}

/**
 * コサイン類似度 (Cosine Similarity)
 * 2つのベクトル間のコサイン類似度 (-1.0 〜 1.0) を計算します。
 *
 * @param {number[] | Float32Array} v1 - 1つ目のベクトル
 * @param {number[] | Float32Array} v2 - 2つ目のベクトル
 * @returns {number} コサイン類似度 (-1.0 〜 1.0)。ゼロベクトルを含む場合は 0 を返します。
 * @throws {Error} ベクトルの次元数が異なる場合にエラーをスローします。
 */
export function cosineSimilarity(
  v1: number[] | Float32Array,
  v2: number[] | Float32Array,
): number {
  const dim = v1.length;
  if (dim !== v2.length) {
    throw new Error("Vectors must have the same dimension.");
  }
  let dot = 0;
  let norm1Sq = 0;
  let norm2Sq = 0;
  for (let i = 0; i < dim; i++) {
    const val1 = v1[i];
    const val2 = v2[i];
    dot += val1 * val2;
    norm1Sq += val1 * val1;
    norm2Sq += val2 * val2;
  }
  if (norm1Sq === 0 || norm2Sq === 0) return 0;
  return dot / (Math.sqrt(norm1Sq) * Math.sqrt(norm2Sq));
}

/**
 * 直交射影による成分除去 (Orthogonal Rejection / Negative Prompting)
 * baseVector から negativeVector の方向成分を完全に除去した新しいベクトルを返します。
 * 数式: v' = v - (v・u / u・u) * u
 *
 * @param {number[] | Float32Array} baseVector - 元のベクトル (v)
 * @param {number[] | Float32Array} negativeVector - 除去したい成分を持つベクトル (u)
 * @returns {Float32Array} 成分が除去された新しいベクトル (v')
 * @throws {Error} ベクトルの次元数が異なる場合にエラーをスローします。
 */
export function reject(
  baseVector: number[] | Float32Array,
  negativeVector: number[] | Float32Array,
): Float32Array {
  const dim = baseVector.length;
  if (dim !== negativeVector.length) {
    throw new Error("Vectors must have the same dimension.");
  }

  let dotVU = 0;
  let dotUU = 0;
  for (let i = 0; i < dim; i++) {
    const u = negativeVector[i];
    dotVU += baseVector[i] * u;
    dotUU += u * u;
  }

  if (dotUU === 0) {
    // negativeVector がゼロベクトルの場合は何も引けないのでそのまま返す
    const result = new Float32Array(dim);
    result.set(baseVector);
    return result;
  }

  const scalar = dotVU / dotUU;
  const result = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    result[i] = baseVector[i] - scalar * negativeVector[i];
  }

  return result;
}

/**
 * 活性化関数の種類を定義します。
 * @typedef {"linear" | "relu" | "sigmoid" | "tanh"} Activation
 */
export type Activation = "linear" | "relu" | "sigmoid" | "tanh";

/**
 * ベクトルに対して非線形活性化関数を適用します (In-place処理)。
 *
 * @param {Float32Array} vector - 活性化関数を適用する対象のベクトル（直接変更されます）
 * @param {Activation} [activation] - 適用する活性化関数の種類（"linear", "relu", "sigmoid", "tanh"）
 * @returns {void}
 */
export function applyActivationToVector(
  vector: Float32Array,
  activation?: Activation,
): void {
  if (!activation || activation === "linear") return;
  const dim = vector.length;
  if (activation === "relu") {
    for (let i = 0; i < dim; i++) {
      if (vector[i] < 0) vector[i] = 0;
    }
  } else if (activation === "sigmoid") {
    for (let i = 0; i < dim; i++) {
      vector[i] = 1 / (1 + Math.exp(-vector[i]));
    }
  } else if (activation === "tanh") {
    for (let i = 0; i < dim; i++) {
      vector[i] = Math.tanh(vector[i]);
    }
  }
}

/**
 * Softmax関数の計算
 * 数値の配列からSoftmax確率分布を計算します（オーバーフロー防止対策済み）。
 *
 * @param {number[]} values - 入力となる数値の配列
 * @returns {number[]} 確率の合計が1.0となるSoftmax関数適用後の配列
 */
export function softmax(values: number[]): number[] {
  if (values.length === 0) return [];
  const max = Math.max(...values);
  let sum = 0;
  const exps = values.map((v) => {
    const e = Math.exp(v - max);
    sum += e;
    return e;
  });
  return exps.map((e) => e / sum);
}

/**
 * 2次元配列（行列）を指定した次元数で1次元のFloat32Arrayにフラット化します。
 *
 * @param {number[][]} matrix - 2次元配列の行列
 * @param {number} rows - 期待される行数
 * @param {number} cols - 期待される列数
 * @param {string} contextName - エラーメッセージ用のコンテキスト名
 * @returns {Float32Array} フラット化された1次元配列
 * @throws {Error} 次元数が一致しない場合にエラーをスローします。
 */
export function flattenMatrix(
  matrix: number[][],
  rows: number,
  cols: number,
  contextName: string = "Matrix",
): Float32Array {
  if (matrix.length !== rows) {
    throw new Error(
      `${contextName}: Row dimension mismatch. Expected ${rows}, got ${matrix.length}.`,
    );
  }
  const flat = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    if (matrix[i].length !== cols) {
      throw new Error(
        `${contextName}: Column dimension mismatch at row ${i}. Expected ${cols}, got ${matrix[i].length}.`,
      );
    }
    for (let j = 0; j < cols; j++) {
      flat[i * cols + j] = matrix[i][j];
    }
  }
  return flat;
}

/**
 * ベクトルの次元数を検証します。
 *
 * @param {number[] | Float32Array} vector - 検証するベクトル
 * @param {number} expectedDimension - 期待される次元数
 * @param {string} contextName - エラーメッセージ用のコンテキスト名
 * @throws {Error} 次元数が一致しない場合にエラーをスローします。
 */
export function assertDimension(
  vector: number[] | Float32Array,
  expectedDimension: number,
  contextName: string = "Vector",
): void {
  if (vector.length !== expectedDimension) {
    throw new Error(
      `${contextName} dimension mismatch. Expected ${expectedDimension}, got ${vector.length}.`,
    );
  }
}
