import {
  WarpAdapter,
  FinalStageAdapter,
  AdapterState,
} from "../interfaces/WarpAdapter";

/**
 * WarpPipeline で利用可能なアダプタの復元用ファクトリ関数を管理するレジストリ。
 */
export class AdapterRegistry {
  private static adapters = new Map<
    string,
    (state: AdapterState) => WarpAdapter
  >();
  private static finalStages = new Map<
    string,
    (state: AdapterState) => FinalStageAdapter
  >();

  public static register(
    type: string,
    importFn: (state: AdapterState) => WarpAdapter,
  ): void {
    AdapterRegistry.adapters.set(type, importFn);
  }

  public static get(
    type: string,
  ): ((state: AdapterState) => WarpAdapter) | undefined {
    return AdapterRegistry.adapters.get(type);
  }

  public static registerFinalStage(
    type: string,
    importFn: (state: AdapterState) => FinalStageAdapter,
  ): void {
    AdapterRegistry.finalStages.set(type, importFn);
  }

  public static getFinalStage(
    type: string,
  ): ((state: AdapterState) => FinalStageAdapter) | undefined {
    return AdapterRegistry.finalStages.get(type);
  }
}
