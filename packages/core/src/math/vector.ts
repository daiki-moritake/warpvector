import { WarpDimensionMismatchError } from "../errors";

/**
 * ベクトルの次元数を検証します。
 *
 * @param {number[] | Float32Array} vector - 検証するベクトル
 * @param {number} expectedDimension - 期待される次元数
 * @param {string} contextName - エラーメッセージ用のコンテキスト名
 * @throws {WarpDimensionMismatchError} 次元数が一致しない場合にエラーをスローします。
 */
export function assertDimension(
  vector: number[] | Float32Array,
  expectedDimension: number,
  contextName: string = "Vector",
): void {
  if (vector.length !== expectedDimension) {
    throw new WarpDimensionMismatchError(
      contextName,
      expectedDimension,
      vector.length,
    );
  }
}

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
 * target ベクトルに対して、source ベクトルを定数倍したものを足し込みます（インプレース操作）。
 * target += scale * source (BLAS の axpy 相当)
 *
 * @param target 更新対象のベクトル (In-place)
 * @param source 足し込むベクトル
 * @param scale 掛け合わせるスカラー値 (デフォルト: 1.0)
 * @throws {Error} ベクトルの次元数が異なる場合にエラーをスローします。
 */
export function addScaledVector(
  target: Float32Array | number[],
  source: Float32Array | number[],
  scale: number = 1.0,
): void {
  const dim = target.length;
  if (dim !== source.length) {
    throw new Error("Vectors must have the same dimension.");
  }
  for (let i = 0; i < dim; i++) {
    target[i] += scale * source[i];
  }
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

// ハミング距離計算用のルックアップテーブル (LUT) を作成
const POPCOUNT_LUT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  let count = 0;
  let n = i;
  while (n > 0) {
    count++;
    n &= n - 1;
  }
  POPCOUNT_LUT[i] = count;
}

/**
 * Binary量子化された2つのベクトル間のハミング距離を計算します。
 * ハミング距離が小さいほど類似度が高いことを意味します。
 */
export function hammingDistance(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) throw new Error("Length mismatch");
  let distance = 0;
  for (let i = 0; i < a.length; i++) {
    distance += POPCOUNT_LUT[a[i] ^ b[i]];
  }
  return distance;
}

/**
 * Int8量子化された2つのベクトル間のドット積（内積）を計算します。
 * 動的スケーリングが埋め込まれている場合はスケールを戻して計算します。
 */
export function int8DotProduct(a: Int8Array, b: Int8Array): number {
  if (a.length !== b.length) throw new Error("Length mismatch");

  // 動的スケーリング埋め込み（dim + 4）かどうかの自動判別
  let isDynamic = false;
  let maxA = 1.0;
  let maxB = 1.0;

  if (a.length > 4) {
    const dim = a.length - 4;
    // 暗黙のtry-catchを避けてDataViewから読み取る。
    // byteLength と length は TypedArray で等しいため範囲外アクセスは起きない。
    const viewA = new DataView(a.buffer, a.byteOffset, a.byteLength);
    const viewB = new DataView(b.buffer, b.byteOffset, b.byteLength);
    maxA = viewA.getFloat32(dim, true);
    maxB = viewB.getFloat32(dim, true);
    
    // 妥当な浮動小数点スケール値であるかの検証
    if (
      Number.isFinite(maxA) &&
      Number.isFinite(maxB) &&
      maxA > 0 &&
      maxA < 1000.0 &&
      maxB > 0 &&
      maxB < 1000.0
    ) {
      isDynamic = true;
    }
  }

  if (isDynamic) {
    const dim = a.length - 4;
    let dot = 0;
    for (let i = 0; i < dim; i++) {
      dot += a[i] * b[i];
    }
    return dot * (maxA / 127.0) * (maxB / 127.0);
  } else {
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }
}

/**
 * Int8量子化された2つのベクトル間のコサイン類似度（-1.0〜1.0）を計算します。
 * 動的スケーリングが埋め込まれている場合は、スケール部分（末尾4バイト）を除外して実データのみで計算します。
 */
export function int8CosineSimilarity(a: Int8Array, b: Int8Array): number {
  if (a.length !== b.length) throw new Error("Length mismatch");

  let isDynamic = false;
  if (a.length > 4) {
    const dim = a.length - 4;
    // 暗黙のtry-catchを避けてDataViewから読み取る。
    const viewA = new DataView(a.buffer, a.byteOffset, a.byteLength);
    const viewB = new DataView(b.buffer, b.byteOffset, b.byteLength);
    const maxA = viewA.getFloat32(dim, true);
    const maxB = viewB.getFloat32(dim, true);
    
    if (
      Number.isFinite(maxA) &&
      Number.isFinite(maxB) &&
      maxA > 0 &&
      maxA < 1000.0 &&
      maxB > 0 &&
      maxB < 1000.0
    ) {
      isDynamic = true;
    }
  }

  const dim = isDynamic ? a.length - 4 : a.length;
  let dot = 0;
  let normASq = 0;
  let normBSq = 0;
  
  for (let i = 0; i < dim; i++) {
    const valA = a[i];
    const valB = b[i];
    dot += valA * valB;
    normASq += valA * valA;
    normBSq += valB * valB;
  }
  
  if (normASq === 0 || normBSq === 0) return 0;
  return dot / (Math.sqrt(normASq) * Math.sqrt(normBSq));
}

/**
 * 2つのベクトル間の類似度スコアを計算します。
 * 全ての型に対して「元のFloat32空間でのコサイン類似度近似値（-1.0 〜 1.0）」という統一規格で返します。
 * 異なるベクトル空間（型）同士の比較はエラーをスローします。
 *
 * - Binary(Uint8Array): ハミング距離からのコサイン近似 (1 - 2*H/dim)
 * - Int8(Int8Array): Int8空間でのコサイン類似度
 * - Float32(Float32Array) / number[]: 通常のコサイン類似度
 */
export function computeVectorScore(a: number[] | Float32Array | Int8Array | Uint8Array, b: number[] | Float32Array | Int8Array | Uint8Array): number {
  const typeA = a instanceof Uint8Array ? "binary" : a instanceof Int8Array ? "int8" : "float";
  const typeB = b instanceof Uint8Array ? "binary" : b instanceof Int8Array ? "int8" : "float";
  
  if (typeA !== typeB) {
    throw new Error(`Cannot compute similarity between different vector types: ${typeA} and ${typeB}`);
  }

  if (a instanceof Uint8Array && b instanceof Uint8Array) {
    const dim = a.length * 8; // ビット数
    const h = hammingDistance(a, b);
    return 1.0 - (2.0 * h) / dim;
  } else if (a instanceof Int8Array && b instanceof Int8Array) {
    return int8CosineSimilarity(a, b);
  } else {
    return cosineSimilarity(
      a as number[] | Float32Array,
      b as number[] | Float32Array
    );
  }
}
