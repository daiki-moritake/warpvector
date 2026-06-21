import {
  assertDimension,
  type WarpAdapter,
  type FinalStageAdapter,
  safeJsonParse,
  assertObject,
  assertPositiveInt,
  assertType,
  getWasmInstance,
  allocateWasmMemory,
  withWasmMemoryStack,
  writeFloat32ArrayToWasm
} from "@warpvector/core";

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

export type QuantizationType = "int8" | "binary";

export interface QuantizationConfig {
  /**
   * "int8": 8-bit スカラー量子化 (Int8Array を返す)
   * "binary": 1-bit 二値化 (Uint8Array を返す、8次元分を1バイトにパック)
   */
  type: QuantizationType;
  /**
   * 量子化するベクトルの次元数
   */
  dim: number;
  /**
   * 動的キャリブレーション（スケール算出）を有効にするか。
   * trueの場合、ベクトルの絶対値の最大値 max(abs(v)) を算出し、
   * scale = 127 / max(abs(v)) を用いて量子化します。
   */
  dynamic?: boolean;
}

/**
 * QuantizationAdapter は、Float32のベクトルを Int8 や Binary に圧縮し、
 * メモリ使用量と保存コストを劇的に（1/4 〜 1/32 に）削減します。
 */
export class QuantizationAdapter implements WarpAdapter, FinalStageAdapter {
  private type: QuantizationType;
  private dim: number;
  private dynamic: boolean;
  private wasm: WebAssembly.Instance | null;

  constructor(config: QuantizationConfig) {
    this.type = config.type;
    this.dim = config.dim;
    this.dynamic = config.dynamic ?? false;
    this.wasm = getWasmInstance();

    if (this.type === "binary" && this.dim % 8 !== 0) {
      throw new Error(
        `Binary quantization requires dimension to be a multiple of 8. Got ${this.dim}`,
      );
    }
  }

  public tune(vector: number[] | Float32Array): Int8Array | Uint8Array {
    assertDimension(vector, this.dim, "QuantizationAdapter.tune");

    if (this.wasm) {
      const exports = this.wasm.exports as any;
      if (exports.quantizeToInt8Wasm && exports.quantizeToBinaryWasm) {
        return this.tuneWasm(vector, exports);
      }
    }

    return this.tuneJs(vector);
  }

  private tuneWasm(vector: number[] | Float32Array, exports: any): Int8Array | Uint8Array {
    const memory = exports.memory as WebAssembly.Memory;

    return withWasmMemoryStack(() => {
      const vectorPtr = allocateWasmMemory(this.dim * 4);
      writeFloat32ArrayToWasm(memory, vector, vectorPtr);

      if (this.type === "int8") {
        const outLen = this.dynamic ? this.dim + 4 : this.dim;
        const outPtr = allocateWasmMemory(outLen);

        exports.quantizeToInt8Wasm(vectorPtr, outPtr, this.dim, this.dynamic);

        const i8view = new Int8Array(memory.buffer, outPtr, outLen);
        const result = new Int8Array(outLen);
        result.set(i8view);
        return result;
      } else {
        const outLen = this.dim / 8;
        const outPtr = allocateWasmMemory(outLen);

        exports.quantizeToBinaryWasm(vectorPtr, outPtr, this.dim);

        const u8view = new Uint8Array(memory.buffer, outPtr, outLen);
        const result = new Uint8Array(outLen);
        result.set(u8view);
        return result;
      }
    });
  }

  private tuneJs(vector: number[] | Float32Array): Int8Array | Uint8Array {
    if (this.type === "int8") {
      if (this.dynamic) {
        // 動的量子化: ベクトルの絶対値の最大値を求める (ゼロ除算防止のため最小閾値 1e-8)
        let maxVal = 1e-8;
        for (let i = 0; i < this.dim; i++) {
          const abs = Math.abs(vector[i]);
          if (abs > maxVal) maxVal = abs;
        }

        const scale = 127.0 / maxVal;
        // 最後に最大値 maxVal (Float32 = 4バイト) を埋め込むため、dim + 4 バイト確保
        const result = new Int8Array(this.dim + 4);
        for (let i = 0; i < this.dim; i++) {
          let val = Math.round(vector[i] * scale);
          if (val > 127) val = 127;
          if (val < -128) val = -128;
          result[i] = val;
        }

        // 最後の 4 バイトに maxVal をリトルエンディアンで書き込む
        const view = new DataView(
          result.buffer,
          result.byteOffset,
          result.byteLength,
        );
        view.setFloat32(this.dim, maxVal, true);
        return result;
      } else {
        // 8-bit Scalar Quantization ([-1.0, 1.0] -> [-127, 127])
        const result = new Int8Array(this.dim);
        for (let i = 0; i < this.dim; i++) {
          // ベクトル要素が正規化されている前提で、スケーリングを行う
          // 範囲外の値は -128 ~ 127 にクリップされる
          let val = Math.round(vector[i] * 127);
          if (val > 127) val = 127;
          if (val < -128) val = -128;
          result[i] = val;
        }
        return result;
      }
    } else if (this.type === "binary") {
      // 1-bit Binary Quantization (ビットパッキング)
      const bytesLength = this.dim >> 3;
      const result = new Uint8Array(bytesLength);

      // 8要素ごとにループ展開して高速化
      for (let i = 0, byteIndex = 0; i < this.dim; i += 8, byteIndex++) {
        let byte = 0;
        if (vector[i] > 0) byte |= 128;     // 1 << 7
        if (vector[i + 1] > 0) byte |= 64;  // 1 << 6
        if (vector[i + 2] > 0) byte |= 32;  // 1 << 5
        if (vector[i + 3] > 0) byte |= 16;  // 1 << 4
        if (vector[i + 4] > 0) byte |= 8;   // 1 << 3
        if (vector[i + 5] > 0) byte |= 4;   // 1 << 2
        if (vector[i + 6] > 0) byte |= 2;   // 1 << 1
        if (vector[i + 7] > 0) byte |= 1;   // 1 << 0
        result[byteIndex] = byte;
      }
      return result;
    }

    throw new Error(`Unknown quantization type: ${this.type}`);
  }

  /**
   * Binary量子化された2つのベクトル間のハミング距離を計算します。
   * ハミング距離が小さいほど類似度が高いことを意味します。
   */
  public static hammingDistance(a: Uint8Array, b: Uint8Array): number {
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
  public static int8DotProduct(a: Int8Array, b: Int8Array): number {
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
   * FinalStageAdapter の実装: Float32Array を量子化します。
   * WarpPipeline.setFinalStage() で使用される場合はこのメソッドが呼ばれます。
   */
  public encode(vector: Float32Array): Int8Array | Uint8Array {
    return this.tune(vector);
  }

  public exportState(): string {
    return JSON.stringify({
      type: this.type,
      dim: this.dim,
      dynamic: this.dynamic,
    });
  }

  public static importState(stateJson: string): QuantizationAdapter {
    const data = assertObject(
      safeJsonParse(stateJson, "QuantizationAdapter"),
      "root",
    );
    assertType(data.type, "string", "type");
    assertPositiveInt(data.dim, "dim");
    return new QuantizationAdapter({
      type: data.type as QuantizationType,
      dim: data.dim as number,
      dynamic: typeof data.dynamic === "boolean" ? data.dynamic : false,
    });
  }
}
