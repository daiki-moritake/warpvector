// @warpvector/extras - Extended adapters for WarpVector
import { WarpPipeline } from "@warpvector/core";
import { QuantizationAdapter } from "./QuantizationAdapter";

export * from "./ColbertAdapter";
export * from "./QuantizationAdapter";
export * from "./VsaAdapter";
export * from "./TaskArithmetic";
export * from "./fusion";

// WarpPipeline にextras系アダプタを自動登録 (side-effect)
WarpPipeline.registerAdapter("QuantizationAdapter", (state) =>
  QuantizationAdapter.importState(state as string),
);
