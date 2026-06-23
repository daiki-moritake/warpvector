/**
 * @warpvector/experimental
 *
 * 実験的な機能のバンドル。APIは安定していないため、
 * メジャーバージョンアップなしに変更される可能性があります。
 *
 * @experimental
 *
 * 含まれる機能:
 * - ColBERT Late Interaction (MaxSim)
 * - Anomaly Detection (Mahalanobis distance)
 * - Vector Symbolic Architecture (VSA)
 * - Task Arithmetic (model merging)
 *
 * @example
 * ```typescript
 * // 実験的機能は明示的にインポート
 * import { ColbertAdapter, VsaAdapter, AnomalyDetectionAdapter } from "@warpvector/experimental";
 * ```
 */

// Re-export experimental features from @warpvector/extras
export { ColbertAdapter } from "@warpvector/extras";
export { VsaAdapter, type VsaConfig } from "@warpvector/extras";
export { AnomalyDetectionAdapter } from "@warpvector/extras";
export {
  TaskArithmetic,
  type TaskVector,
  type MergeStrategy,
} from "@warpvector/extras";
