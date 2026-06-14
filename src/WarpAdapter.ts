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
   * @returns 変換後のベクトル (Float32Array)
   */
  tune(vector: number[] | Float32Array, context?: string): Float32Array;
}
