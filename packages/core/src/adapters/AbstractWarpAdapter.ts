import { WarpAdapter, InputVector, TransformOutput, AdapterState } from "../interfaces/WarpAdapter";

/**
 * WarpAdapter の基本実装を提供する抽象クラス。
 * tuneBatch のデフォルト実装などを提供します。
 */
export abstract class AbstractWarpAdapter implements WarpAdapter {
  public abstract tune(vector: InputVector, context?: string): TransformOutput;

  public tuneBatch(vectors: InputVector[], context?: string): TransformOutput[] {
    return vectors.map(v => this.tune(v, context));
  }

  public async tuneBatchAsync(vectors: InputVector[], context?: string): Promise<TransformOutput[]> {
    // デフォルト実装は同期版を Promise でラップするだけ
    return Promise.resolve(this.tuneBatch(vectors, context));
  }

  public async init?(): Promise<void> {
    // デフォルトは何もしない
    return Promise.resolve();
  }

  public exportState?(): AdapterState {
    throw new Error("exportState is not implemented for this adapter.");
  }
}
