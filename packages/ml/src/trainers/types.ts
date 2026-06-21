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
}
