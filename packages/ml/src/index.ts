// @warpvector/ml - Lightweight non-linear and online learning adapters for edge inference
import { WarpPipeline } from "@warpvector/core";
import { MlpAdapter } from "./adapters/MlpAdapter";
import { WhiteningAdapter } from "./adapters/WhiteningAdapter";
import { MoeAdapter } from "./adapters/MoeAdapter";

export * from "./adapters/MoeAdapter";
export * from "./adapters/MlpAdapter";
export * from "./adapters/WhiteningAdapter";

// WarpPipeline にml系アダプタを自動登録 (side-effect)
WarpPipeline.registerAdapter("MlpAdapter", (state) =>
  MlpAdapter.importState(state as any),
);
WarpPipeline.registerAdapter("WhiteningAdapter", (state) =>
  WhiteningAdapter.importState(state as any),
);
WarpPipeline.registerAdapter("MoeAdapter", (state) =>
  MoeAdapter.importState(state as any),
);
