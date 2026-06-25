// @warpvector/extras - Extended adapters for WarpVector
import { WarpPipeline } from "@warpvector/core";
import {
  QuantizationAdapter,
  QuantizationConfig,
} from "./adapters/QuantizationAdapter";
import { AnomalyDetectionAdapter } from "./adapters/AnomalyDetectionAdapter";
import { SafeQuantizationAdapter } from "./adapters/SafeQuantizationAdapter";

export * from "./adapters/ColbertAdapter";
export * from "./adapters/QuantizationAdapter";
export * from "./adapters/SafeQuantizationAdapter";
export * from "./adapters/AnomalyDetectionAdapter";
export * from "./adapters/VsaAdapter";
export * from "./operations/TaskArithmetic";
export * from "./fusion";

/**
 * WarpPipeline にワンライナーで量子化 FinalStage を設定するヘルパー関数。
 *
 * @example
 * import { quantizePipeline } from "@warpvector/extras";
 *
 * const pipeline = quantizePipeline(
 *   new WarpPipeline(768).addIntent({ ... }),
 *   { type: "int8", dim: 768 }
 * );
 *
 * @param pipeline 量子化を追加するパイプライン
 * @param config 量子化設定
 * @returns 量子化が設定されたパイプライン
 */
export function quantizePipeline(
  pipeline: WarpPipeline,
  config: QuantizationConfig,
): WarpPipeline {
  const quantizer = new QuantizationAdapter(config);
  return pipeline.setFinalStage("QuantizationAdapter", quantizer);
}

// WarpPipeline にextras系アダプタを自動登録 (side-effect)
WarpPipeline.registerAdapter("QuantizationAdapter", (state) =>
  QuantizationAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("AnomalyDetectionAdapter", (state) =>
  AnomalyDetectionAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("SafeQuantizationAdapter", (state) =>
  SafeQuantizationAdapter.importState(state as string),
);

// FinalStageAdapter としても登録（パイプライン末尾の量子化復元用）
WarpPipeline.registerFinalStage("QuantizationAdapter", (state) =>
  QuantizationAdapter.importState(state as string),
);

