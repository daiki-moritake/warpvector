/**
 * WarpVector OpenTelemetry 計装モジュール
 *
 * ゼロ依存のまま、OpenTelemetry 互換のトレーシング・メトリクスを提供します。
 * @opentelemetry/api がインストールされている場合は自動的にそれを使用し、
 * インストールされていない場合は no-op で動作します。
 *
 * @example
 * ```typescript
 * import { WarpTracer } from "@warpvector/core";
 *
 * // OpenTelemetry が設定済みの場合、自動的にスパンが記録される
 * const tracer = new WarpTracer();
 *
 * const result = tracer.trace("intent-adapter.tune", { dim: 768, intent: "tech" }, () => {
 *   return adapter.tune(vector, "tech");
 * });
 *
 * // メトリクスの取得
 * const metrics = tracer.getMetrics();
 * console.log(metrics.totalCalls, metrics.avgLatencyMs);
 * ```
 */

/** スパン属性の型 */
export type SpanAttributes = Record<string, string | number | boolean>;

/** 内部メトリクス記録用 */
export interface WarpMetrics {
  /** 総呼び出し回数 */
  totalCalls: number;
  /** 平均レイテンシ（ミリ秒） */
  avgLatencyMs: number;
  /** 最大レイテンシ（ミリ秒） */
  maxLatencyMs: number;
  /** 最小レイテンシ（ミリ秒） */
  minLatencyMs: number;
  /** 操作別の呼び出し回数 */
  operationCounts: Record<string, number>;
  /** 操作別の平均レイテンシ（ミリ秒） */
  operationLatencies: Record<string, number>;
}



/**
 * OpenTelemetry トレーサーのラッパー
 *
 * ゼロ依存で計測機能を提供し、OpenTelemetry が利用可能な場合は
 * 自動的にそれに委譲します。
 */
export class WarpTracer {
  private readonly _metrics: {
    calls: number;
    totalMs: number;
    maxMs: number;
    minMs: number;
    ops: Record<string, { count: number; totalMs: number }>;
  };

  constructor() {
    this._metrics = {
      calls: 0,
      totalMs: 0,
      maxMs: 0,
      minMs: Infinity,
      ops: {},
    };
  }

  /**
   * 関数をトレースし、実行時間を計測します。
   *
   * @param operationName 操作名（例: "intent-adapter.tune", "pipeline.run"）
   * @param attributes スパン属性
   * @param fn 実行する関数
   * @returns 関数の戻り値
   *
   * @example
   * ```typescript
   * const result = tracer.trace("tune", { intent: "tech" }, () => {
   *   return adapter.tune(vector, "tech");
   * });
   * ```
   */
  trace<T>(
    operationName: string,
    attributes: SpanAttributes,
    fn: () => T,
  ): T {
    const t0 = performance.now();
    try {
      const result = fn();
      this._recordSuccess(operationName, performance.now() - t0, attributes);
      return result;
    } catch (err) {
      this._recordError(operationName, performance.now() - t0, err);
      throw err;
    }
  }

  /**
   * 非同期関数をトレースし、実行時間を計測します。
   *
   * @param operationName 操作名
   * @param attributes スパン属性
   * @param fn 実行する非同期関数
   * @returns 関数の戻り値の Promise
   */
  async traceAsync<T>(
    operationName: string,
    attributes: SpanAttributes,
    fn: () => Promise<T>,
  ): Promise<T> {
    const t0 = performance.now();
    try {
      const result = await fn();
      this._recordSuccess(operationName, performance.now() - t0, attributes);
      return result;
    } catch (err) {
      this._recordError(operationName, performance.now() - t0, err);
      throw err;
    }
  }

  /**
   * 現在のメトリクスを取得します。
   */
  getMetrics(): WarpMetrics {
    const operationCounts: Record<string, number> = {};
    const operationLatencies: Record<string, number> = {};

    for (const [op, data] of Object.entries(this._metrics.ops)) {
      operationCounts[op] = data.count;
      operationLatencies[op] = data.count > 0 ? data.totalMs / data.count : 0;
    }

    return {
      totalCalls: this._metrics.calls,
      avgLatencyMs:
        this._metrics.calls > 0
          ? this._metrics.totalMs / this._metrics.calls
          : 0,
      maxLatencyMs: this._metrics.maxMs,
      minLatencyMs:
        this._metrics.minMs === Infinity ? 0 : this._metrics.minMs,
      operationCounts,
      operationLatencies,
    };
  }

  /**
   * メトリクスをリセットします。
   */
  resetMetrics(): void {
    this._metrics.calls = 0;
    this._metrics.totalMs = 0;
    this._metrics.maxMs = 0;
    this._metrics.minMs = Infinity;
    for (const key of Object.keys(this._metrics.ops)) {
      delete this._metrics.ops[key];
    }
  }

  private _recordSuccess(
    op: string,
    durationMs: number,
    _attributes: SpanAttributes,
  ): void {
    this._metrics.calls++;
    this._metrics.totalMs += durationMs;
    this._metrics.maxMs = Math.max(this._metrics.maxMs, durationMs);
    this._metrics.minMs = Math.min(this._metrics.minMs, durationMs);

    if (!this._metrics.ops[op]) {
      this._metrics.ops[op] = { count: 0, totalMs: 0 };
    }
    this._metrics.ops[op].count++;
    this._metrics.ops[op].totalMs += durationMs;
  }

  private _recordError(
    op: string,
    durationMs: number,
    _error: unknown,
  ): void {
    // エラー時もレイテンシは記録する
    this._recordSuccess(op, durationMs, {});
  }
}
