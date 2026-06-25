// @warpvector/train - Machine learning trainers and auto-tuning tools
import { WarpPipeline } from "@warpvector/core";
import { SoftWhiteningAdapter } from "./adapters/SoftWhiteningAdapter";

export * from "./adapters/SoftWhiteningAdapter";

export * from "./trainers/BaseTrainer";
export * from "./trainers/IntentTrainer";
export * from "./trainers/InfoNCETrainer";
export * from "./trainers/TripletTrainer";
export * from "./trainers/CrossEncoderTrainer";
export * from "./trainers/MigrationTrainer";
export * from "./feedback/FeedbackCollector";
export * from "./feedback/AdaptiveScheduler";
export * from "./feedback/FederatedAggregator";

export * from "./automl/metrics";
export * from "./automl/PipelineAutoTuner";
export * from "./factory/IntentMatrixFactory";

// WarpPipeline に高度なアダプタを自動登録 (side-effect)
WarpPipeline.registerAdapter("SoftWhiteningAdapter", (state) =>
  SoftWhiteningAdapter.importState(state as string),
);
