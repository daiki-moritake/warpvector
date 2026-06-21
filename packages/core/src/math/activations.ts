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
  let max = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }
  let sum = 0;
  const exps = values.map((v) => {
    const e = Math.exp(v - max);
    sum += e;
    return e;
  });
  return exps.map((e) => e / sum);
}
