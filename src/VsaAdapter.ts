import { assertDimension, normalize } from "./utils";

/**
 * ベクトル・シンボリック・アーキテクチャ (VSA) / 超次元計算アダプタ
 *
 * ベクトル同士を論理的・数学的に結合（バインド）したり束ねたり（バンドル）することで、
 * 1つの密なベクトルの中にキーと値（メタデータなど）を埋め込み、検索空間上で
 * そのまま演算を行えるようにする機能を提供します。
 */
export class VsaAdapter {
  /**
   * ベクトルのバンドリング (Bundling / Superposition)
   * 複数のベクトルを足し合わせ（重ね合わせ）て1つのベクトルに統合します。
   * 「A と B の両方の概念を含む」ベクトルを作成する際に使用します。
   *
   * @param vectors 束ねるベクトルの配列
   * @param shouldNormalize 結果をL2正規化するかどうか（デフォルト: true）
   * @returns 束ねられた新しいベクトル
   */
  public static bundle(
    vectors: (number[] | Float32Array)[],
    shouldNormalize: boolean = true
  ): Float32Array {
    if (vectors.length === 0) {
      throw new Error("Cannot bundle an empty array of vectors.");
    }
    
    const dim = vectors[0].length;
    const result = new Float32Array(dim);

    for (let i = 0; i < vectors.length; i++) {
      const vec = vectors[i];
      assertDimension(vec, dim, `Vector at index ${i}`);
      for (let j = 0; j < dim; j++) {
        result[j] += vec[j];
      }
    }

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
   * @param shouldNormalize 結果をL2正規化するかどうか（デフォルト: true）
   * @returns バインドされた新しいベクトル
   */
  public static bind(
    vec1: number[] | Float32Array,
    vec2: number[] | Float32Array,
    shouldNormalize: boolean = true
  ): Float32Array {
    const dim = vec1.length;
    assertDimension(vec2, dim, "Vector 2");

    const result = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      result[i] = vec1[i] * vec2[i];
    }

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
   * @param shouldNormalize 結果をL2正規化するかどうか（デフォルト: true）
   * @returns アンバインドされて抽出されたベクトル
   */
  public static unbind(
    boundVec: number[] | Float32Array,
    keyVec: number[] | Float32Array,
    shouldNormalize: boolean = true
  ): Float32Array {
    const dim = boundVec.length;
    assertDimension(keyVec, dim, "Key Vector");

    const result = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      // ゼロ除算を防ぐための微小なイプシロン
      const val = keyVec[i] === 0 ? 1e-8 : keyVec[i];
      result[i] = boundVec[i] / val;
    }

    if (shouldNormalize) {
      return normalize(result);
    }

    return result;
  }
}
