export type InputVector = number[] | Float32Array;
export type OutputVector = Float32Array | Int8Array | Uint8Array;
export type AdapterState = Record<string, unknown> | string;

/**
 * WarpVector のすべてのベクトル変換アダプターに共通するインターフェース。
 * 外部の統合ライブラリ（Prisma, LangChainなど）は、このインターフェースを通じて
 * さまざまなアダプター（IntentAdapter, MlpAdapter, WhiteningAdapter など）を
 * 透過的に扱うことができます。
 */
export interface WarpAdapter {
  /**
   * 与えられたベクトルを変換（ワープ）します。
   * 
   * @param vector 変換前のベクトル (number[] または Float32Array)
   * @param context オプションのコンテキスト情報 (意図の名前、バージョンなど)
   * @returns 変換後のベクトル (Float32Array, 量子化の場合は Int8Array や Uint8Array)
   */
  tune(vector: InputVector, context?: string): OutputVector;

  /**
   * 複数のベクトルを一括で変換します（オプション実装）
   * WASMやSIMDを使用した最適化処理を提供します。
   * 
   * @param vectors 変換前のベクトルの配列
   * @param context オプションのコンテキスト情報
   */
  tuneBatch?(vectors: InputVector[], context?: string): OutputVector[];

  /**
   * 非同期での初期化処理（オプション実装）
   * WASMのロードなどが必要なアダプタで実装します。
   */
  init?(): Promise<void>;

  /**
   * アダプタの状態（学習済み重みなど）をエクスポートします（オプション実装）
   * importState に渡して完全に復元可能なJSON形式を返します。
   */
  exportState?(): AdapterState;
}
