/**
 * @warpvector/experimental
 *
 * @deprecated このパッケージは非推奨です。v0.9.0 以降、すべての機能は
 * 安定版パッケージから直接インポートしてください:
 *
 * - `ColbertAdapter` → `@warpvector/rerank` または `warpvector/rerank`
 * - `VsaAdapter` → `@warpvector/extras` または `warpvector/extras`
 * - `AnomalyDetectionAdapter` → `@warpvector/extras` または `warpvector/extras`
 * - `TaskArithmetic` → `@warpvector/extras` または `warpvector/extras`
 *
 * また、`warpvector` のルートインポートからもすべて利用可能です:
 * ```typescript
 * import { ColbertAdapter, VsaAdapter, AnomalyDetectionAdapter, TaskArithmetic } from "warpvector";
 * ```
 *
 * このパッケージは互換性のため re-export を維持していますが、
 * 次のメジャーバージョン (1.0.0) で完全に削除される予定です。
 *
 * @example
 * ```typescript
 * // ❌ 非推奨: @warpvector/experimental からのインポート
 * import { ColbertAdapter } from "@warpvector/experimental";
 *
 * // ✅ 推奨: 安定版パッケージからのインポート
 * import { ColbertAdapter } from "warpvector/rerank";
 * // or
 * import { ColbertAdapter } from "warpvector";
 * ```
 */

// Runtime deprecation warning (emitted once)
let _warned = false;
function _emitDeprecationWarning(): void {
  if (!_warned) {
    _warned = true;
    console.warn(
      "[@warpvector/experimental] This package is deprecated. " +
        "Import directly from stable packages instead:\n" +
        '  - ColbertAdapter        → "warpvector/rerank" or "warpvector"\n' +
        '  - VsaAdapter            → "warpvector/extras" or "warpvector"\n' +
        '  - AnomalyDetectionAdapter → "warpvector/extras" or "warpvector"\n' +
        '  - TaskArithmetic        → "warpvector/extras" or "warpvector"\n' +
        "This package will be removed in v1.0.0.",
    );
  }
}
_emitDeprecationWarning();

// Re-export experimental features from @warpvector/extras and @warpvector/rerank
export { ColbertAdapter } from "@warpvector/rerank";
export { VsaAdapter, type VsaOptions } from "@warpvector/extras";
export { AnomalyDetectionAdapter } from "@warpvector/extras";
export { TaskArithmetic, type TaskConfig } from "@warpvector/extras";
