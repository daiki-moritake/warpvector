import {
  allocateWasmMemory,
  assertDimension,
  getWasmInstance,
  withWasmMemoryStack,
  writeFloat32ArrayToWasm,
} from "@warpvector/core";

/**
 * Adam最適化のステート変数を管理する共通基底クラス
 */
export abstract class AbstractAdamTrainer<TResult = import("@warpvector/core").IntentWeights> {
  protected t: number = 0;
  protected mW!: Float32Array;
  protected vW!: Float32Array;
  protected mb!: Float32Array;
  protected vb!: Float32Array;

  protected initAdamState(sDim: number, tDim: number): void {
    if (!this.mW || this.mW.length !== sDim * tDim) {
      this.mW = new Float32Array(sDim * tDim);
      this.vW = new Float32Array(sDim * tDim);
      this.mb = new Float32Array(tDim);
      this.vb = new Float32Array(tDim);
      this.t = 0;
    }
    assertDimension(this.mW, sDim * tDim, "AdamState mW");
  }

  /**
   * 学習済みの1次元行列とバイアスから最終的な重みデータを構築します。
   * デフォルトでは IntentWeights または互換性のある型にキャストして返します。
   * @param flatMatrix 学習済みのフラット化された行列
   * @param bias 学習済みのバイアスベクトル
   * @returns TResult
   */
  protected toWeights(
    flatMatrix: Float32Array,
    bias: Float32Array,
  ): TResult {
    return {
      matrix: flatMatrix,
      bias: bias,
    } as unknown as TResult;
  }

  /**
   * アフィンレイヤー (matrix, bias) に対する Adam のパラメータ更新を適用します。
   * InfoNCE や Triplet など、様々な損失関数で計算された勾配(outputGradients)を元に更新を行います。
   */
  protected applyAdamToAffine(
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

        const adamUpdateWasm = instance.exports.adamUpdateWasm as CallableFunction;
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
        mMatrix.set(f32.subarray(mMatrixPtr / 4, mMatrixPtr / 4 + mMatrix.length));
        vMatrix.set(f32.subarray(vMatrixPtr / 4, vMatrixPtr / 4 + vMatrix.length));
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
}
