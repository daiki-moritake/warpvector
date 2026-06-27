import { assertDimension, flattenMatrix, applyAffine } from "../utils";
import {
  safeJsonParse,
  assertPositiveInt,
  assertObject,
  assertNumberArray,
} from "../validation";
import { ProjectionAdapter, ProjectionWeights } from "./ProjectionAdapter";

/**
 * AlignmentAdapter クラス
 * 古い埋め込みモデルから新しいモデルへの移行や、別言語へのマッピングなど、
 * 異なるベクトル空間同士のアラインメント（空間翻訳）を行うためのアダプタです。
 * 内部的には ProjectionAdapter の機能（アフィン変換）を利用し、ドメインに特化したインターフェースを提供します。
 */
export class AlignmentAdapter extends ProjectionAdapter {
  /**
   * AlignmentAdapter を初期化します。
   *
   * @constructor
   * @param {number} sourceDimension - 変換元の空間の次元数（例: 古いモデルの次元数）
   * @param {number} targetDimension - 変換先の空間の次元数（例: 新しいモデルの次元数）
   * @param {Record<string, ProjectionWeights>} [alignments] - 初期化時に追加するアラインメント設定
   */
  constructor(
    sourceDimension: number,
    targetDimension: number,
    alignments?: Record<string, ProjectionWeights>,
  ) {
    super(sourceDimension, targetDimension, alignments);
  }

  /**
   * 新しいアラインメント（空間変換行列）を追加します。
   *
   * @param {string} name - アラインメント名（例: "v1_to_v2", "en_to_ja"）
   * @param {ProjectionWeights} weights - 空間変換行列とバイアス
   */
  public addAlignment(name: string, weights: ProjectionWeights): void {
    this.addProjection(name, weights);
  }

  /**
   * 変換元のベクトルを、対象のベクトル空間へと翻訳（アライン）します。
   *
   * @param vector 変換元のベクトル
   * @param targetSpace アラインメント先の設定名
   * @returns 変換先の空間におけるベクトル
   */
  public align(
    vector: number[] | Float32Array,
    targetSpace: string = "default",
  ): Float32Array {
    return this.tune(vector, targetSpace);
  }

  /**
   * エクスポートされた状態から AlignmentAdapter を復元します。
   */
  public static importState(stateJson: string): AlignmentAdapter {
    const data = assertObject(
      safeJsonParse(stateJson, "AlignmentAdapter"),
      "root",
    );
    const inDim = assertPositiveInt(data.inDimension, "inDimension");
    const outDim = assertPositiveInt(data.outDimension, "outDimension");
    const adapter = new AlignmentAdapter(inDim, outDim);

    const projections = assertObject(data.projections, "projections");
    for (const [name, rawProj] of Object.entries(projections)) {
      const proj = assertObject(rawProj, `projections.${name}`);
      const matrix = assertNumberArray(
        proj.matrix,
        `projections.${name}.matrix`,
      );
      
      const weights: ProjectionWeights = { matrix: new Float32Array(matrix) };
      if (proj.bias) {
        weights.bias = new Float32Array(assertNumberArray(proj.bias, `projections.${name}.bias`));
      }
      adapter.addAlignment(name, weights);
    }
    return adapter;
  }
}
