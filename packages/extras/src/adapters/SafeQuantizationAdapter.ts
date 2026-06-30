import {
  type FinalStageAdapter,
  type OutputVector,
  safeJsonParse,
  assertObject,
  assertPositiveInt,
  assertType,
  type AdapterState,
} from "@warpvector/core";
import { QuantizationAdapter, QuantizationConfig } from "./QuantizationAdapter";

export interface SafeQuantizationOptions extends QuantizationConfig {
  /**
   * 量子化前にベクトルの値をクランプ（上限・下限）する際の最大絶対値。
   * 指定しない場合、動的に最大値を探索するか、型の限界（int8なら127など）が適用されますが、
   * 安全のため明示的に指定することを推奨します。
   */
  clipThreshold?: number;
}

/**
 * QuantizationAdapter の安全なラッパー。
 * オーバーフローやNaNによって量子化後の値が破壊されるのを防ぐため、
 * 事前に厳密なサニタイズとクリッピングを行います。
 */
export class SafeQuantizationAdapter implements FinalStageAdapter {
  private baseAdapter: QuantizationAdapter;
  private options: SafeQuantizationOptions;

  constructor(options: SafeQuantizationOptions) {
    if (arguments.length > 1) {
      throw new Error(
        "[WarpVector DX Error] SafeQuantizationAdapter のコンストラクタ引数が変更されました。\n" +
          "次元数 (dim) などを第1引数に渡す必要はありません。すべての設定は1つのオブジェクトで渡してください。\n" +
          "例: new SafeQuantizationAdapter({ type: 'int8', dim: 1536, dynamic: true })",
      );
    }

    // 古いプロパティ名への警告
    if (options && "levels" in options) {
      throw new Error(
        "[WarpVector DX Error] SafeQuantizationAdapter のプロパティ 'levels' は廃止されました。\n" +
          "'type': 'int8' または 'binary' を使用してください。",
      );
    }
    if (options && "adaptiveRange" in options) {
      throw new Error(
        "[WarpVector DX Error] SafeQuantizationAdapter のプロパティ 'adaptiveRange' は 'dynamic' に変更されました。\n" +
          "例: new SafeQuantizationAdapter({ type: 'int8', dim: 1536, dynamic: true })",
      );
    }

    if (options && options.clipThreshold !== undefined) {
      if (
        typeof options.clipThreshold !== "number" ||
        options.clipThreshold <= 0 ||
        Number.isNaN(options.clipThreshold)
      ) {
        throw new Error(
          "[WarpVector DX Error] SafeQuantizationAdapter の 'clipThreshold' は正の数値でなければなりません。",
        );
      }
    }

    this.options = options;
    this.baseAdapter = new QuantizationAdapter(options);
  }

  public async init(): Promise<void> {
    // QuantizationAdapter は初期化不要
  }

  public tune(vector: Float32Array): Float32Array {
    // No-op for quantization
    return vector;
  }

  public encode(vector: Float32Array): OutputVector {
    const len = vector.length;
    const safeVector = new Float32Array(len);

    // safe threshold for specific quantization types
    let defaultClip = 100.0;
    if (this.options.type === "int8") {
      defaultClip = 127.0; // int8 の最大値
    } else if (this.options.type === "binary") {
      defaultClip = Infinity; // binaryは符号のみ見るのでクリップの影響は少ない
    }

    const clipThreshold = this.options.clipThreshold ?? defaultClip;

    // 1st pass: NaN/Infinity check and find max absolute value
    let maxAbs = 0.0;
    for (let i = 0; i < len; i++) {
      const val = vector[i];
      if (Number.isNaN(val) || !Number.isFinite(val)) {
        throw new Error(
          `[WarpVector Error] SafeQuantizationAdapter encountered an invalid value (${val}) at index ${i}. ` +
            "Quantization aborted to prevent data corruption.",
        );
      }
      const absVal = Math.abs(val);
      if (absVal > maxAbs) {
        maxAbs = absVal;
      }
    }

    // Determine scale factor to preserve vector direction
    const scale = maxAbs > clipThreshold ? clipThreshold / maxAbs : 1.0;

    // 2nd pass: apply scaling and copy
    for (let i = 0; i < len; i++) {
      safeVector[i] = vector[i] * scale;
    }

    // サニタイズされた安全なベクトルを本家のQuantizationAdapterに渡す
    return this.baseAdapter.encode(safeVector);
  }

  public exportState(): AdapterState {
    return {
      __version: "1.0",
      ...this.options,
    };
  }

  public static importState(data: AdapterState): SafeQuantizationAdapter {
    assertType(data.type, "string", "type");
    const dim = assertPositiveInt(data.dim, "dim");

    const options: SafeQuantizationOptions = {
      type: data.type as any,
      dim: dim,
      dynamic: typeof data.dynamic === "boolean" ? data.dynamic : false,
      clipThreshold:
        typeof data.clipThreshold === "number" ? data.clipThreshold : undefined,
    };
    return new SafeQuantizationAdapter(options);
  }
}
