import { OutputVector } from "../interfaces/WarpAdapter";
import { FormatOptions } from "./WarpPipeline";

/**
 * ベクトルデータベース等の出力フォーマット変換を管理するレジストリ。
 */
export class FormatRegistry {
  private static formats = new Map<
    string,
    (vector: OutputVector, options: FormatOptions) => unknown
  >();

  public static register(
    format: string,
    formatFn: (vector: OutputVector, options: FormatOptions) => unknown,
  ): void {
    FormatRegistry.formats.set(format, formatFn);
  }

  public static get(
    format: string,
  ): ((vector: OutputVector, options: FormatOptions) => unknown) | undefined {
    return FormatRegistry.formats.get(format);
  }
}
