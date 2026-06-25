import {
  getWasmInstance,
  allocateWasmMemory,
  writeFloat32ArrayToWasm,
  withWasmMemoryStack,
} from "@warpvector/core";

/**
 * アフィンレイヤー (matrix, bias) に対する Adam のパラメータ更新を適用します。
 * 損失関数で計算された勾配(outputGradients)を元に In-place で更新を行います。
 *
 * WASMがロードされていればWASMで高速に処理し、そうでない場合はJSのフォールバックを使用します。
 *
 * @param matrix 重み行列 (更新対象)
 * @param bias バイアスベクトル (更新対象)
 * @param mMatrix 重み行列のモメンタム (1次モーメント)
 * @param vMatrix 重み行列の速度 (2次モーメント)
 * @param mBias バイアスのモメンタム
 * @param vBias バイアスの速度
 * @param input 入力ベクトル
 * @param outputGradients 出力層の勾配
 * @param lr 学習率
 * @param reg L2正則化係数
 * @param t 現在のステップ数 (1から始まる)
 */
export function applyAdamToAffine(
  matrix: Float32Array,
  bias: Float32Array,
  mMatrix: Float32Array,
  vMatrix: Float32Array,
  mBias: Float32Array,
  vBias: Float32Array,
  input: number[] | Float32Array,
  outputGradients: number[] | Float32Array,
  lr: number,
  reg: number,
  t: number,
): void {
  const tDim = bias.length;
  const sDim = input.length;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 1e-8;

  const instance = getWasmInstance();
  if (instance && instance.exports.adamUpdateWasm) {
    const memory = instance.exports.memory as WebAssembly.Memory;

    const matrixBytes = matrix.byteLength;
    const biasBytes = bias.byteLength;
    const inputBytes = sDim * 4;
    const gradBytes = tDim * 4;

    withWasmMemoryStack(() => {
      const matrixPtr = allocateWasmMemory(matrixBytes);
      const biasPtr = allocateWasmMemory(biasBytes);
      const mMatrixPtr = allocateWasmMemory(matrixBytes);
      const vMatrixPtr = allocateWasmMemory(matrixBytes);
      const mBiasPtr = allocateWasmMemory(biasBytes);
      const vBiasPtr = allocateWasmMemory(biasBytes);
      const inputPtr = allocateWasmMemory(inputBytes);
      const gradPtr = allocateWasmMemory(gradBytes);

      // データの書き込み
      writeFloat32ArrayToWasm(memory, matrix, matrixPtr);
      writeFloat32ArrayToWasm(memory, bias, biasPtr);
      writeFloat32ArrayToWasm(memory, mMatrix, mMatrixPtr);
      writeFloat32ArrayToWasm(memory, vMatrix, vMatrixPtr);
      writeFloat32ArrayToWasm(memory, mBias, mBiasPtr);
      writeFloat32ArrayToWasm(memory, vBias, vBiasPtr);
      writeFloat32ArrayToWasm(memory, input, inputPtr);
      writeFloat32ArrayToWasm(memory, outputGradients, gradPtr);

      const adamUpdateWasm = instance.exports
        .adamUpdateWasm as CallableFunction;
      adamUpdateWasm(
        matrixPtr,
        biasPtr,
        mMatrixPtr,
        vMatrixPtr,
        mBiasPtr,
        vBiasPtr,
        inputPtr,
        gradPtr,
        lr,
        reg,
        beta1,
        beta2,
        epsilon,
        t,
        sDim,
        tDim,
      );

      // 更新されたデータを読み戻す
      const f32 = new Float32Array(memory.buffer);
      matrix.set(f32.subarray(matrixPtr / 4, matrixPtr / 4 + matrix.length));
      bias.set(f32.subarray(biasPtr / 4, biasPtr / 4 + bias.length));
      mMatrix.set(
        f32.subarray(mMatrixPtr / 4, mMatrixPtr / 4 + mMatrix.length),
      );
      vMatrix.set(
        f32.subarray(vMatrixPtr / 4, vMatrixPtr / 4 + vMatrix.length),
      );
      mBias.set(f32.subarray(mBiasPtr / 4, mBiasPtr / 4 + mBias.length));
      vBias.set(f32.subarray(vBiasPtr / 4, vBiasPtr / 4 + vBias.length));
    });
  } else {
    // WASMが使えない場合のJSフォールバック
    for (let i = 0; i < tDim; i++) {
      const bGrad = outputGradients[i];

      // Adam for Bias
      mBias[i] = beta1 * mBias[i] + (1 - beta1) * bGrad;
      vBias[i] = beta2 * vBias[i] + (1 - beta2) * (bGrad * bGrad);
      const mHatB = mBias[i] / (1 - Math.pow(beta1, t));
      const vHatB = vBias[i] / (1 - Math.pow(beta2, t));

      bias[i] -= (lr * mHatB) / (Math.sqrt(vHatB) + epsilon);

      const rowOffset = i * sDim;
      for (let j = 0; j < sDim; j++) {
        const wIdx = rowOffset + j;
        const wGrad = bGrad * input[j] + reg * matrix[wIdx];

        mMatrix[wIdx] = beta1 * mMatrix[wIdx] + (1 - beta1) * wGrad;
        vMatrix[wIdx] = beta2 * vMatrix[wIdx] + (1 - beta2) * (wGrad * wGrad);
        const mHatW = mMatrix[wIdx] / (1 - Math.pow(beta1, t));
        const vHatW = vMatrix[wIdx] / (1 - Math.pow(beta2, t));

        matrix[wIdx] -= (lr * mHatW) / (Math.sqrt(vHatW) + epsilon);
      }
    }
  }
}
