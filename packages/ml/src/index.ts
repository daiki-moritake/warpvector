// @warpvector/ml - Machine learning adapters and trainers
import { WarpPipeline } from "@warpvector/core";
import { MlpAdapter } from "./adapters/MlpAdapter";
import { WhiteningAdapter } from "./adapters/WhiteningAdapter";
import { SoftWhiteningAdapter } from "./adapters/SoftWhiteningAdapter";
import { MoeAdapter } from "./adapters/MoeAdapter";

export * from "./adapters/MoeAdapter";
export * from "./adapters/MlpAdapter";
export * from "./adapters/WhiteningAdapter";
export * from "./adapters/SoftWhiteningAdapter";
export * from "./rerankers/TimeReversalReranker";
export * from "./rerankers/MultipathScatteringReranker";
export * from "./rerankers/BaseGraphReranker";
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
