import {
  type WarpAdapter,
  type InputVector,
  type TransformOutput,
  assertDimension,
  assertArray,
  AbstractWarpAdapter,
  safeJsonParse,
  assertObject,
} from "@warpvector/core";

export interface AnomalyDetectionConfig {
  /**
   * 動作モード。
   * 'strict': 異常を検知した場合にエラーをスローする。
   * 'safe': 異常を検知した場合、クリッピングやゼロ埋めを行って安全なベクトルとして通す。
   * デフォルトは 'strict'
   */
  mode?: "strict" | "safe";

  /**
   * ベクトルの要素の絶対値の許容最大値。
   * これを超える値は異常とみなされます。デフォルトは 100.0。
   */
  maxValue?: number;
}

/**
 * 外部から入力されるベクトルに異常（NaN, Infinity, 極端な外れ値）がないかを
 * 検知・サニタイズするためのセキュリティ・アダプタ。
 */
export class AnomalyDetectionAdapter extends AbstractWarpAdapter {
  private mode: "strict" | "safe";
  private maxValue: number;

  constructor(config: AnomalyDetectionConfig = {}) {
    super();
    if (arguments.length > 1) {
      throw new Error(
        "[WarpVector DX Error] AnomalyDetectionAdapter のコンストラクタ引数が変更されました。\n" +
          "次元数 (dim) などを第1引数に渡す必要はありません。すべての設定は1つのオブジェクトで渡してください。\n" +
          "例: new AnomalyDetectionAdapter({ mode: 'safe', maxValue: 3.0 })",
      );
    }

    // threshold のような古いプロパティ名への警告
    if (config && "threshold" in config) {
      throw new Error(
        "[WarpVector DX Error] AnomalyDetectionAdapter のプロパティ 'threshold' は 'maxValue' に変更されました。\n" +
          "例: new AnomalyDetectionAdapter({ maxValue: 3.0 })",
      );
    }

    if (config.maxValue !== undefined) {
      if (
        typeof config.maxValue !== "number" ||
        config.maxValue <= 0 ||
        Number.isNaN(config.maxValue)
      ) {
        throw new Error(
          "[WarpVector DX Error] AnomalyDetectionAdapter の 'maxValue' は正の数値でなければなりません。",
        );
      }
    }

    this.mode = config.mode || "strict";
    this.maxValue = config.maxValue || 100.0;
  }

  public async init(): Promise<void> {
    // 初期化処理は不要
  }

  public tune(vector: InputVector): TransformOutput {
    let hasAnomaly = false;
    let anomalyReason = "";

    for (let i = 0; i < vector.length; i++) {
      const val = vector[i];
      if (Number.isNaN(val)) {
        hasAnomaly = true;
        anomalyReason = `NaN detected at index ${i}`;
        break;
      }
      if (!Number.isFinite(val)) {
        hasAnomaly = true;
        anomalyReason = `Infinity detected at index ${i}`;
        break;
      }
      if (Math.abs(val) > this.maxValue) {
        hasAnomaly = true;
        anomalyReason = `Value ${val} exceeds maxValue ${this.maxValue} at index ${i}`;
        break;
      }
    }

    if (!hasAnomaly) {
      return new Float32Array(vector);
    }

    if (this.mode === "strict") {
      throw new Error(
        `AnomalyDetectionAdapter [STRICT MODE]: ${anomalyReason}`,
      );
    }

    // safe mode: NaN/Infinityはエラーにし、外れ値は全体スケーリングで対応
    let maxAbs = 0.0;
    for (let i = 0; i < vector.length; i++) {
      const val = vector[i];
      if (Number.isNaN(val) || !Number.isFinite(val)) {
        throw new Error(
          `AnomalyDetectionAdapter [SAFE MODE]: Invalid value (${val}) detected at index ${i}. Processing aborted.`
        );
      }
      const absVal = Math.abs(val);
      if (absVal > maxAbs) {
        maxAbs = absVal;
      }
    }

    const scale = maxAbs > this.maxValue ? this.maxValue / maxAbs : 1.0;
    const safeVector = new Float32Array(vector.length);
    
    for (let i = 0; i < vector.length; i++) {
      safeVector[i] = vector[i] * scale;
    }

    return safeVector;
  }

  public exportState(): string {
    return JSON.stringify({
      __version: "1.0",
      mode: this.mode,
      maxValue: this.maxValue,
    });
  }

  public static importState(stateJson: string): AnomalyDetectionAdapter {
    const data = assertObject(
      safeJsonParse(stateJson, "AnomalyDetectionAdapter"),
      "root",
    );
    const mode =
      data.mode === "strict" || data.mode === "safe" ? data.mode : "strict";
    const maxValue =
      typeof data.maxValue === "number" ? data.maxValue : 100.0;
    return new AnomalyDetectionAdapter({ mode, maxValue });
  }
}
