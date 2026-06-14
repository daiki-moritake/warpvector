import { assertDimension, normalize, addScaledVector } from "./utils";

/**
 * ベクトル・シンボリック・アーキテクチャ (VSA) / 超次元計算アダプタ
 *
 * ベクトル同士を論理的・数学的に結合（バインド）したり束ねたり（バンドル）することで、
 * 1つの密なベクトルの中にキーと値（メタデータなど）を埋め込み、検索空間上で
 * そのまま演算を行えるようにする機能を提供します。
 */

/**
 * VSA演算のオプション
 */
export interface VsaOptions {
  /** 結果をL2正規化するかどうか（デフォルト: true） */
  shouldNormalize?: boolean;
}

export class VsaAdapter {
  /**
   * ベクトルのバンドリング (Bundling / Superposition)
   * 複数のベクトルを足し合わせ（重ね合わせ）て1つのベクトルに統合します。
   * 「A と B の両方の概念を含む」ベクトルを作成する際に使用します。
   *
   * @param vectors 束ねるベクトルの配列
   * @param options 演算オプション
   * @returns 束ねられた新しいベクトル
   */
  public static bundle(
    vectors: (number[] | Float32Array)[],
    options: VsaOptions = {},
  ): Float32Array {
    if (vectors.length === 0) {
      throw new Error("Cannot bundle an empty array of vectors.");
    }

    const dim = vectors[0].length;
    const result = new Float32Array(dim);

    for (let i = 0; i < vectors.length; i++) {
      const vec = vectors[i];
      assertDimension(vec, dim, `Vector at index ${i}`);
      addScaledVector(result, vec, 1.0);
    }

    const shouldNormalize = options.shouldNormalize ?? true;
    if (shouldNormalize) {
      return normalize(result);
    }

    return result;
  }

  /**
   * ベクトルのバインディング (Binding / Hadamard Product)
   * アダマール積（要素ごとの積）を用いて、2つのベクトルを「結合」します。
   * 例: キー（ユーザーID）と値（好み）を掛け合わせ、特有の「ユーザーの好み」ベクトルを生成します。
   *
   * @param vec1 バインドするベクトル1
   * @param vec2 バインドするベクトル2
   * @param options 演算オプション
   * @returns バインドされた新しいベクトル
   */
  public static bind(
    vec1: number[] | Float32Array,
    vec2: number[] | Float32Array,
    options: VsaOptions = {},
  ): Float32Array {
    const dim = vec1.length;
    assertDimension(vec2, dim, "Vector 2");

    const result = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      result[i] = vec1[i] * vec2[i];
    }

    const shouldNormalize = options.shouldNormalize ?? true;
    if (shouldNormalize) {
      return normalize(result);
    }

    return result;
  }

  /**
   * ベクトルのアンバインディング (Unbinding)
   * バインドされたベクトルから、片方のベクトル（キー）を使って元の値（バリュー）を取り出します。
   * アダマール積によるバインディングの逆演算（要素ごとの除算）を行います。
   *
   * @param boundVec バインド済みのベクトル
   * @param keyVec 抽出に使用するキーベクトル
   * @param options 演算オプション
   * @returns アンバインドされて抽出されたベクトル
   */
  public static unbind(
    boundVec: number[] | Float32Array,
    keyVec: number[] | Float32Array,
    options: VsaOptions = {},
  ): Float32Array {
    const dim = boundVec.length;
    assertDimension(keyVec, dim, "Key Vector");

    const result = new Float32Array(dim);
    const EPSILON = 1e-5;
    for (let i = 0; i < dim; i++) {
      let val = keyVec[i];
      if (Math.abs(val) < EPSILON) {
        val = val >= 0 ? EPSILON : -EPSILON;
      }
      result[i] = boundVec[i] / val;
    }

    const shouldNormalize = options.shouldNormalize ?? true;
    if (shouldNormalize) {
      return normalize(result);
    }

    return result;
  }

  /**
   * ---------------------------------------------------------
   * Binary VSA (バイナリベクトル・シンボリック・アーキテクチャ)
   * ---------------------------------------------------------
   * QuantizationAdapter で 1-bit (Binary) 量子化された Uint8Array ベクトルに
   * 対する超次元計算を行います。XOR 演算により、超高速・極小メモリでの処理が可能です。
   */

  /**
   * バイナリベクトルのバインディング (Binary Binding / XOR)
   * XOR (排他的論理和) を用いて2つのバイナリベクトルを結合します。
   * Binary VSA において、XOR は情報を結合するための標準的な演算です。
   *
   * @param bin1 バインドするバイナリベクトル1 (Uint8Array)
   * @param bin2 バインドするバイナリベクトル2 (Uint8Array)
   * @returns バインドされた新しいバイナリベクトル (Uint8Array)
   */
  public static bindBinary(bin1: Uint8Array, bin2: Uint8Array): Uint8Array {
    if (bin1.length !== bin2.length) {
      throw new Error("Binary vectors must have the same length in bytes.");
    }
    const len = bin1.length;
    const result = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      result[i] = bin1[i] ^ bin2[i]; // XOR演算
    }
    return result;
  }

  /**
   * バイナリベクトルのアンバインディング (Binary Unbinding / XOR)
   * XOR の自己逆性 (A ^ B ^ B = A) を利用して、キーを用いて元の値を抽出します。
   * 内部的には bindBinary と全く同じ処理です。
   *
   * @param boundBin バインド済みのバイナリベクトル (Uint8Array)
   * @param keyBin 抽出に使用するキーバイナリベクトル (Uint8Array)
   * @returns アンバインドされて抽出されたバイナリベクトル (Uint8Array)
   */
  public static unbindBinary(
    boundBin: Uint8Array,
    keyBin: Uint8Array,
  ): Uint8Array {
    return VsaAdapter.bindBinary(boundBin, keyBin);
  }

  /**
   * バイナリベクトルのバンドリング (Binary Bundling / Majority Vote)
   * 複数のバイナリベクトルを重ね合わせます。各ビット位置で 1 と 0 の出現回数をカウントし、
   * 多数決 (Majority Vote) で最終的なビットを決定します。
   *
   * @param bins 束ねるバイナリベクトルの配列 (Uint8Arrayの配列)
   * @returns 束ねられた新しいバイナリベクトル (Uint8Array)
   */
  public static bundleBinary(bins: Uint8Array[]): Uint8Array {
    if (bins.length === 0) {
      throw new Error("Cannot bundle an empty array of binary vectors.");
    }

    const numVectors = bins.length;
    const len = bins[0].length;
    const result = new Uint8Array(len);

    for (let i = 0; i < len; i++) {
      let resultByte = 0;
      // 各バイトの 8 つのビット(0~7)について多数決をとる
      for (let bit = 0; bit < 8; bit++) {
        let onesCount = 0;
        const mask = 1 << bit;

        for (let v = 0; v < numVectors; v++) {
          if (bins[v].length !== len) {
            throw new Error(
              `Binary vector at index ${v} has mismatched length.`,
            );
          }
          if ((bins[v][i] & mask) !== 0) {
            onesCount++;
          }
        }

        // 多数決 (半数より多ければ 1 を立てる。同数の場合は 0 とするが、ランダムでもよい)
        if (onesCount > numVectors / 2) {
          resultByte |= mask;
        } else if (onesCount === numVectors / 2) {
          // タイブレーク: 最適化のため、単に 0 または 1 に固定する(ここでは1とする)
          // 完全なランダムタイブレークを実装することも可能だが、パフォーマンス優先で固定する。
          resultByte |= mask;
        }
      }
      result[i] = resultByte;
    }

    return result;
  }
}
