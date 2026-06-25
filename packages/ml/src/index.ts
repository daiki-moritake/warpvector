// @warpvector/ml - Lightweight non-linear and online learning adapters for edge inference
import { WarpPipeline } from "@warpvector/core";
import { MlpAdapter } from "./adapters/MlpAdapter";
import { WhiteningAdapter } from "./adapters/WhiteningAdapter";
import { SoftWhiteningAdapter } from "./adapters/SoftWhiteningAdapter";
import { MoeAdapter } from "./adapters/MoeAdapter";

export * from "./adapters/MoeAdapter";
export * from "./adapters/MlpAdapter";
export * from "./adapters/WhiteningAdapter";
export * from "./adapters/SoftWhiteningAdapter";

// WarpPipeline にml系アダプタを自動登録 (side-effect)
WarpPipeline.registerAdapter("MlpAdapter", (state) =>
  MlpAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("WhiteningAdapter", (state) =>
  WhiteningAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("SoftWhiteningAdapter", (state) =>
  SoftWhiteningAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("MoeAdapter", (state) =>
  MoeAdapter.importState(state as string),
);
