import type { IntentWeights } from "@warpvector/core";

/**
 * AdaptiveScheduler が受け付けるオンライン学習トレーナーのインターフェース。
 * TripletTrainer と InfoNCETrainer の両方がこのインターフェースを満たします。
 */
export interface OnlineTrainer<TExample> {
  updateOnline(
    currentWeights: IntentWeights,
    example: TExample,
    options?: { learningRate?: number },
  ): Promise<IntentWeights>;
}

/**
 * AdaptiveScheduler のオプション。
 */
export interface SchedulerOptions {
  /** 初期学習率。デフォルト: 0.01 */
  initialLearningRate?: number;
  /** 最小学習率（これ以下には減衰しない）。デフォルト: 0.0001 */
  minLearningRate?: number;
  /** 減衰率。lr(n) = max(min, initial / (1 + decayRate * n))。デフォルト: 0.001 */
  decayRate?: number;
  /** バッファにこの件数溜まったら自動学習。デフォルト: 5 */
  batchSize?: number;
  /** バッファの最大サイズ。デフォルト: 100 */
  maxBufferSize?: number;
}

/**
 * AdaptiveScheduler のシリアライズ状態。
 */
interface SchedulerState {
  totalSteps: number;
  initialLearningRate: number;
  minLearningRate: number;
  decayRate: number;
  batchSize: number;
  maxBufferSize: number;
}

/**
 * 学習率の自動減衰とバッチ学習のタイミング制御を行うスケジューラー。
 *
 * ユーザーが使い込むほど学習率が自動的に減衰し、
 * 初期は大きく空間を変形、後期は微調整モードに移行します。
 *
 * @template TExample 学習データの型（TripletExample または InfoNCEExample）
 *
 * @example
 * ```typescript
 * const trainer = new TripletTrainer(1536);
 * const scheduler = new AdaptiveScheduler(trainer, {
 *   initialLearningRate: 0.01,
 *   batchSize: 5,
 * });
 *
 * // フィードバックを追加（5件溜まったら自動で学習）
 * const updated = await scheduler.addFeedback(currentWeights, newExamples);
 * if (updated) {
 *   currentWeights = updated; // 学習が実行された
 * }
 * ```
 */
export class AdaptiveScheduler<TExample> {
  private trainer: OnlineTrainer<TExample>;
  private buffer: TExample[] = [];
  private _totalSteps = 0;

  private readonly initialLearningRate: number;
  private readonly minLearningRate: number;
  private readonly decayRate: number;
  private readonly batchSize: number;
  private readonly maxBufferSize: number;

  constructor(
    trainer: OnlineTrainer<TExample>,
    options: SchedulerOptions = {},
  ) {
    this.trainer = trainer;
    this.initialLearningRate = options.initialLearningRate ?? 0.01;
    this.minLearningRate = options.minLearningRate ?? 0.0001;
    this.decayRate = options.decayRate ?? 0.001;
    this.batchSize = options.batchSize ?? 5;
    this.maxBufferSize = options.maxBufferSize ?? 100;
  }

  /**
   * フィードバックをバッファに追加し、batchSize に達した場合に自動学習を実行します。
   *
   * @param currentWeights 現在の重み
   * @param examples 学習用データの配列
   * @returns 学習が実行された場合は更新後の重み。バッファが batchSize 未満の場合は null。
   */
  public async addFeedback(
    currentWeights: IntentWeights,
    examples: TExample[],
  ): Promise<IntentWeights | null> {
    // バッファに追加
    this.buffer.push(...examples);

    // 上限を超えたら古いものを破棄
    if (this.buffer.length > this.maxBufferSize) {
      this.buffer = this.buffer.slice(-this.maxBufferSize);
    }

    // batchSize 未満なら学習しない
    if (this.buffer.length < this.batchSize) {
      return null;
    }

    let weights = currentWeights;

    // バッファにバッチサイズ以上のデータがある限り、繰り返し学習を行う
    while (this.buffer.length >= this.batchSize) {
      const batch = this.buffer.splice(0, this.batchSize);
      for (const example of batch) {
        weights = await this.trainer.updateOnline(weights, example, {
          learningRate: this.currentLearningRate,
        });
        this._totalSteps++;
      }
    }

    return weights;
  }

  /**
   * バッファの残りを強制的に学習します（batchSize 未満でも実行）。
   *
   * @param currentWeights 現在の重み
   * @returns 学習が実行された場合は更新後の重み。バッファが空の場合は null。
   */
  public async flushAndTrain(
    currentWeights: IntentWeights,
  ): Promise<IntentWeights | null> {
    if (this.buffer.length === 0) {
      return null;
    }

    const batch = this.buffer.splice(0, this.buffer.length);
    let weights = currentWeights;

    for (const example of batch) {
      weights = await this.trainer.updateOnline(weights, example, {
        learningRate: this.currentLearningRate,
      });
      this._totalSteps++;
    }

    return weights;
  }

  /**
   * 現在の学習率（減衰込み）。
   *
   * 計算式: lr(n) = max(minLR, initialLR / (1 + decayRate * n))
   */
  public get currentLearningRate(): number {
    return Math.max(
      this.minLearningRate,
      this.initialLearningRate / (1 + this.decayRate * this._totalSteps),
    );
  }

  /**
   * 累計学習ステップ数。
   */
  public get totalSteps(): number {
    return this._totalSteps;
  }

  /**
   * 現在バッファに蓄積されている学習データ数。
   */
  public get bufferedCount(): number {
    return this.buffer.length;
  }

  /**
   * スケジューラーの状態をエクスポートします。
   * totalSteps とハイパーパラメータを保存し、復元時に学習率の継続性を保ちます。
   */
  public exportState(): string {
    const state: SchedulerState = {
      totalSteps: this._totalSteps,
      initialLearningRate: this.initialLearningRate,
      minLearningRate: this.minLearningRate,
      decayRate: this.decayRate,
      batchSize: this.batchSize,
      maxBufferSize: this.maxBufferSize,
    };
    return JSON.stringify(state);
  }

  /**
   * エクスポートされた状態からスケジューラーを復元します。
   *
   * @param trainer 復元先の Trainer インスタンス
   * @param json exportState() で得られた JSON 文字列
   */
  public static importState<T>(
    trainer: OnlineTrainer<T>,
    json: string,
  ): AdaptiveScheduler<T> {
    const state: SchedulerState = JSON.parse(json);
    const scheduler = new AdaptiveScheduler(trainer, {
      initialLearningRate: state.initialLearningRate,
      minLearningRate: state.minLearningRate,
      decayRate: state.decayRate,
      batchSize: state.batchSize,
      maxBufferSize: state.maxBufferSize,
    });
    scheduler._totalSteps = state.totalSteps;
    return scheduler;
  }
}
