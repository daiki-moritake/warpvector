import { ProjectionWeights } from "@warpvector/core";
import { BaseTrainer } from "../trainers/BaseTrainer";

/**
 * 移行用の学習データペア（旧モデルのベクトル -> 新モデルのベクトル）
 * @interface MigrationExample
 */
export interface MigrationExample {
  /** 移行元のベクトル (例: ada-002の出力) */
  source: number[] | Float32Array;
  /** 移行先のベクトル (例: text-embedding-3-smallの出力) */
  target: number[] | Float32Array;
}

/**
 * 異なる埋め込みモデル間（例：1536次元から512次元へ）の
 * ベクトル空間を翻訳する行列 (ProjectionWeights) を自動学習するトレーナークラス。
 *
 * 次元数の異なるベクトル表現をマッピングするための射影行列を学習します。
 *
 * @example
 * const trainer = new MigrationTrainer(1536, 512);
 * trainer.addExample({ source: [...], target: [...] });
 * const projectionWeights = await trainer.train({ epochs: 200 });
 */
export class MigrationTrainer extends BaseTrainer<
  MigrationExample,
  ProjectionWeights
> {
  private _sourceDimension: number;
  private _targetDimension: number;

  /**
   * @param {number} sourceDimension 移行元の次元数 (例: ada-002なら1536)
   * @param {number} targetDimension 移行先の次元数 (例: text-embedding-3-smallなら512)
   */
  constructor(sourceDimension: number, targetDimension: number) {
    super();
    this._sourceDimension = sourceDimension;
    this._targetDimension = targetDimension;
  }

  protected get sourceDimension(): number {
    return this._sourceDimension;
  }

  protected get targetDimension(): number {
    return this._targetDimension;
  }

  protected getInputs(example: MigrationExample): {
    source: number[] | Float32Array;
    target: number[] | Float32Array;
  } {
    return { source: example.source, target: example.target };
  }
}
