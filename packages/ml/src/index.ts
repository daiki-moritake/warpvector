// @warpvector/ml - Machine learning adapters and trainers
import { WarpPipeline } from "@warpvector/core";
import { MlpAdapter } from "./MlpAdapter";
import { WhiteningAdapter } from "./WhiteningAdapter";

export * from "./MlpAdapter";
export * from "./WhiteningAdapter";
export * from "./BaseTrainer";
export * from "./trainer";
export * from "./InfoNCETrainer";
export * from "./TripletTrainer";
export * from "./migration";

// WarpPipeline にml系アダプタを自動登録 (side-effect)
WarpPipeline.registerAdapter("MlpAdapter", (state) =>
  MlpAdapter.importState(state as string),
);
WarpPipeline.registerAdapter("WhiteningAdapter", (state) =>
  WhiteningAdapter.importState(state as string),
);
