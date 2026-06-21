import { WarpAdapter, InputVector, OutputVector } from "@warpvector/core";

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
export class AnomalyDetectionAdapter implements WarpAdapter {
  private mode: "strict" | "safe";
  private maxValue: number;

  constructor(config: AnomalyDetectionConfig = {}) {
    this.mode = config.mode || "strict";
    this.maxValue = config.maxValue || 100.0;
  }

  public async init(): Promise<void> {
    // 初期化処理は不要
  }

  public tune(vector: InputVector): OutputVector {
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
      throw new Error(`AnomalyDetectionAdapter [STRICT MODE]: ${anomalyReason}`);
    }

    // safe mode: クリップ処理とNaN/Infinityのゼロ埋め
    const safeVector = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
      let val = vector[i];
      if (Number.isNaN(val) || !Number.isFinite(val)) {
        val = 0.0;
      } else if (val > this.maxValue) {
        val = this.maxValue;
      } else if (val < -this.maxValue) {
        val = -this.maxValue;
      }
      safeVector[i] = val;
    }

    return safeVector;
  }

  public exportState(): string {
    return JSON.stringify({ mode: this.mode, maxValue: this.maxValue });
  }

  public static importState(state: string): AnomalyDetectionAdapter {
    const config = JSON.parse(state);
    return new AnomalyDetectionAdapter(config);
  }
}
