/**
 * エポックごとのメトリクス情報。
 * onEpochEnd コールバックに渡されます。
 */
export interface EpochMetrics {
  /** 学習用データの平均ロス */
  trainLoss: number;
  /** 検証用データの平均ロス（validationSplit が設定されている場合のみ） */
  valLoss?: number;
}

/**
 * 基本的な学習オプションを定義するインターフェース。
 * 勾配降下法における各種ハイパーパラメータを設定します。
 */
export interface BaseTrainingOptions {
  /** 学習率 (Learning Rate)。1ステップで重みをどれだけ更新するか。デフォルト: 0.01 */
  learningRate?: number;
  /** 学習のエポック数 (データセット全体を何回繰り返して学習するか)。デフォルト: 100 */
  epochs?: number;
  /** L2正則化の強さ。過学習を防ぐために使用します。デフォルト: 0.001 */
  regularization?: number;
  /** trueの場合、事前に数エポックのテストランを行い、最適な学習率を自動探索します。デフォルト: false */
  autoTune?: boolean;
  /** Lossの改善が見られなかった場合に、学習を早期終了 (Early Stopping) するまでのエポック数 (例: 5) */
  patience?: number;
  /** Early Stopping 発動時に呼ばれるコールバック。指定しない場合、サイレントに停止します。 */
  onEarlyStopping?: (epoch: number, patience: number) => void;

  /**
   * 検証用データの割合（0.0 〜 1.0）。
   * 設定すると、学習データの一部を検証用に分割し、
   * 過学習の検知に使用します。
   * 検証データは学習には使用されません。
   * @example
   * ```typescript
   * const weights = await trainer.train({
   *   validationSplit: 0.2,  // 20%を検証用に分割
   *   patience: 10,          // 検証ロスが10エポック改善しなければ停止
   * });
   * ```
   * @default undefined (分割なし、全データで学習)
   */
  validationSplit?: number;

  /**
   * 各エポック終了時に呼ばれるコールバック。
   * 学習の進捗をモニタリングしたり、ロギングに使用します。
   *
   * @param epoch 現在のエポック番号（1-indexed）
   * @param metrics 学習ロスと検証ロス
   */
  onEpochEnd?: (epoch: number, metrics: EpochMetrics) => void;
}
