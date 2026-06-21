/**
 * WarpVector 軽量メトリクス収集モジュール
 *
 * パイプラインのパフォーマンスを計測するためのゼロ依存メトリクス収集機構です。
 * デフォルトでは無効状態で、`enable()` を呼ぶことで計測を開始します。
 * 無効時はほぼゼロオーバーヘッドです。
 *
 * @example
 * ```typescript
 * const pipeline = new WarpPipeline(1536);
 * pipeline.metrics.enable();
 *
 * pipeline.run(vector, { intent: "tech" });
 *
 * const stats = pipeline.metrics.getMetrics();
 * console.log(`Total runs: ${stats.totalRuns}`);
 * console.log(`Avg duration: ${stats.avgRunDurationMs.toFixed(2)}ms`);
 * ```
 */

/**
 * 個別ステップの計測結果
 */
export interface StepTiming {
  /** ステップの型名（例: "MlpAdapter"） */
  stepType: string;
  /** ステップの実行時間（ミリ秒） */
  durationMs: number;
}

/**
 * パイプライン全体のメトリクスサマリー
 */
export interface PipelineMetrics {
  /** run() の呼び出し回数 */
  totalRuns: number;
  /** runBatch() の呼び出し回数 */
  totalBatchRuns: number;
  /** 処理されたベクトルの総数 */
  totalVectorsProcessed: number;
  /** 各ステップの平均実行時間（ミリ秒） */
  avgStepDurationMs: Record<string, number>;
  /** 直近の run() の実行時間（ミリ秒） */
  lastRunDurationMs: number;
  /** run() の平均実行時間（ミリ秒） */
  avgRunDurationMs: number;
}

/**
 * パイプラインのパフォーマンスメトリクスを収集するコレクター。
 *
 * デフォルトでは無効状態（ゼロオーバーヘッド）で動作し、
 * `enable()` を呼ぶことで計測を開始します。
 */
export class MetricsCollector {
  private _enabled: boolean = false;
  private _totalRuns: number = 0;
  private _totalBatchRuns: number = 0;
  private _totalVectorsProcessed: number = 0;
  private _lastRunDurationMs: number = 0;
  private _totalRunDurationMs: number = 0;
  private _stepDurations: Map<string, { total: number; count: number }> =
    new Map();

  /**
   * メトリクス収集が有効かどうかを返します。
   */
  get enabled(): boolean {
    return this._enabled;
  }

  /**
   * メトリクス収集を有効にします。
   */
  enable(): void {
    this._enabled = true;
  }

  /**
   * メトリクス収集を無効にします。
   * 既に収集済みのメトリクスはリセットされません。
   */
  disable(): void {
    this._enabled = false;
  }

  /**
   * run() の実行を記録します。
   * @returns 計測用のストップウォッチ関数。呼び出すと経過時間を記録します。
   */
  startRun(): (() => void) | null {
    if (!this._enabled) return null;

    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      this._totalRuns++;
      this._totalVectorsProcessed++;
      this._lastRunDurationMs = duration;
      this._totalRunDurationMs += duration;
    };
  }

  /**
   * runBatch() の実行を記録します。
   * @param batchSize バッチ内のベクトル数
   * @returns 計測用のストップウォッチ関数
   */
  startBatchRun(batchSize: number): (() => void) | null {
    if (!this._enabled) return null;

    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      this._totalBatchRuns++;
      this._totalVectorsProcessed += batchSize;
      this._lastRunDurationMs = duration;
      this._totalRunDurationMs += duration;
    };
  }

  /**
   * 個別ステップの実行を記録します。
   * @param stepType ステップの型名
   * @returns 計測用のストップウォッチ関数
   */
  startStep(stepType: string): (() => void) | null {
    if (!this._enabled) return null;

    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      const existing = this._stepDurations.get(stepType);
      if (existing) {
        existing.total += duration;
        existing.count++;
      } else {
        this._stepDurations.set(stepType, { total: duration, count: 1 });
      }
    };
  }

  /**
   * 収集されたメトリクスのサマリーを取得します。
   */
  getMetrics(): PipelineMetrics {
    const avgStepDurationMs: Record<string, number> = {};
    for (const [key, value] of this._stepDurations) {
      avgStepDurationMs[key] = value.count > 0 ? value.total / value.count : 0;
    }

    return {
      totalRuns: this._totalRuns,
      totalBatchRuns: this._totalBatchRuns,
      totalVectorsProcessed: this._totalVectorsProcessed,
      avgStepDurationMs,
      lastRunDurationMs: this._lastRunDurationMs,
      avgRunDurationMs:
        this._totalRuns > 0 ? this._totalRunDurationMs / this._totalRuns : 0,
    };
  }

  /**
   * すべてのメトリクスをリセットします。
   */
  reset(): void {
    this._totalRuns = 0;
    this._totalBatchRuns = 0;
    this._totalVectorsProcessed = 0;
    this._lastRunDurationMs = 0;
    this._totalRunDurationMs = 0;
    this._stepDurations.clear();
  }
}
