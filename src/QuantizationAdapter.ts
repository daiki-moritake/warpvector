import { assertDimension } from "./utils";
import { WarpAdapter } from "./WarpAdapter";

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
}

/**
 * QuantizationAdapter は、Float32のベクトルを Int8 や Binary に圧縮し、
 * メモリ使用量と保存コストを劇的に（1/4 〜 1/32 に）削減します。
 */
export class QuantizationAdapter implements WarpAdapter {
  private type: QuantizationType;
  private dim: number;

  constructor(config: QuantizationConfig) {
    this.type = config.type;
    this.dim = config.dim;
    
    if (this.type === "binary" && this.dim % 8 !== 0) {
      throw new Error(`Binary quantization requires dimension to be a multiple of 8. Got ${this.dim}`);
    }
  }

  public tune(vector: number[] | Float32Array): Int8Array | Uint8Array {
    assertDimension(vector, this.dim, "QuantizationAdapter.tune");

    if (this.type === "int8") {
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

    } else if (this.type === "binary") {
      // 1-bit Binary Quantization (ビットパッキング)
      const bytesLength = this.dim / 8;
      const result = new Uint8Array(bytesLength);
      
      for (let i = 0; i < this.dim; i++) {
        // 値が 0 より大きければ 1、それ以外は 0
        if (vector[i] > 0) {
          const byteIndex = Math.floor(i / 8);
          const bitIndex = i % 8;
          result[byteIndex] |= (1 << (7 - bitIndex));
        }
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
      let xor = a[i] ^ b[i];
      // ブライアン・カーニハンアルゴリズムで1のビット数を数える
      while (xor > 0) {
        distance++;
        xor &= (xor - 1);
      }
    }
    return distance;
  }

  /**
   * Int8量子化された2つのベクトル間のドット積（内積）を計算します。
   */
  public static int8DotProduct(a: Int8Array, b: Int8Array): number {
    if (a.length !== b.length) throw new Error("Length mismatch");
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  public exportState(): string {
    return JSON.stringify({ type: this.type, dim: this.dim });
  }

  public static importState(stateJson: string): QuantizationAdapter {
    const config = JSON.parse(stateJson);
    return new QuantizationAdapter(config);
  }
}
