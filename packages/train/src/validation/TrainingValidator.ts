import { svd, conditionNumber, spectralNorm, identityDeviation } from "../math/svd";

/**
 * 学習結果の検証レポート。
 */
export interface ValidationReport {
  /**
   * 条件数 (σ_max / σ_min)。
   * 高すぎると数値的に不安定。
   * - < 10: 非常に安定
   * - 10–100: 正常
   * - > 100: 警告（過度な変形の可能性）
   * - > 1000: 危険
   */
  conditionNumber: number;

  /**
   * スペクトルノルム (σ_max)。
   * 入力の微小な差が出力でどれだけ増幅されるかを示す。
   * - 1.0 前後: 正常（距離を保存）
   * - > 5: 警告（過度な増幅）
   * - > 10: 危険
   */
  spectralNorm: number;

  /**
   * 単位行列からの乖離度 (‖W - I‖_F / √d)。
   * 0 に近いほど保守的な変換。
   * - < 0.1: 非常に保守的
   * - 0.1–0.5: 正常
   * - > 1.0: 大きな空間変形
   */
  identityDeviation: number;

  /**
   * 行列のランク不足度 (rank(W) / d)。
   * 低すぎると情報が失われている。
   * - 1.0: フルランク
   * - < 0.5: 警告（次元の半分以上が退化）
   */
  rankRatio: number;

  /**
   * 総合判定
   * - "GOOD": 全てのチェックが正常
   * - "WARNING": 一部のチェックが警告レベル
   * - "REJECT": 使用を推奨しない
   */
  verdict: "GOOD" | "WARNING" | "REJECT";

  /** 警告メッセージの配列 */
  warnings: string[];
}

/**
 * 検証オプション。
 */
export interface ValidationOptions {
  /**
   * SVD で計算する特異値の数。
   * 全特異値を計算するとコストが高いため、上位 k 個に制限します。
   * 0 を指定すると min(d, 64) が使用されます。
   * @default 0 (自動)
   */
  numSingularValues?: number;

  /** 条件数の警告閾値 @default 100 */
  conditionNumberWarning?: number;
  /** 条件数の拒否閾値 @default 1000 */
  conditionNumberReject?: number;

  /** スペクトルノルムの警告閾値 @default 5 */
  spectralNormWarning?: number;
  /** スペクトルノルムの拒否閾値 @default 10 */
  spectralNormReject?: number;

  /** 単位行列乖離度の警告閾値 @default 0.5 */
  identityDeviationWarning?: number;
  /** 単位行列乖離度の拒否閾値 @default 2.0 */
  identityDeviationReject?: number;

  /** ランク比率の警告閾値 @default 0.5 */
  rankRatioWarning?: number;
}

/**
 * 学習済みの変換行列の品質を自動検証するバリデーター。
 *
 * 行列の数値的安定性（条件数）、変形の大きさ（スペクトルノルム）、
 * 情報の保存度（ランク）を分析し、本番デプロイ前の品質ゲートとして機能します。
 *
 * @example
 * ```typescript
 * import { TrainingValidator } from "warpvector/train";
 *
 * const validator = new TrainingValidator(1536);
 * const report = validator.validate(learnedWeights.matrix);
 *
 * if (report.verdict === "REJECT") {
 *   console.error("行列の品質が低いです:", report.warnings);
 * }
 * ```
 */
export class TrainingValidator {
  private readonly dimension: number;

  constructor(dimension: number) {
    if (dimension <= 0 || !Number.isInteger(dimension)) {
      throw new Error(
        `Dimension must be a positive integer, got ${dimension}.`,
      );
    }
    this.dimension = dimension;
  }

  /**
   * 変換行列を検証し、品質レポートを生成します。
   *
   * @param matrix 検証対象の変換行列 (d x d, row-major flat Float32Array)
   * @param options 検証オプション
   * @returns 検証レポート
   */
  public validate(
    matrix: Float32Array,
    options: ValidationOptions = {},
  ): ValidationReport {
    const d = this.dimension;
    const expectedSize = d * d;

    if (matrix.length !== expectedSize) {
      throw new Error(
        `Matrix size mismatch: expected ${expectedSize} (${d}x${d}), got ${matrix.length}.`,
      );
    }

    const warnings: string[] = [];
    let hasReject = false;

    // 閾値
    const cnWarn = options.conditionNumberWarning ?? 100;
    const cnReject = options.conditionNumberReject ?? 1000;
    const snWarn = options.spectralNormWarning ?? 5;
    const snReject = options.spectralNormReject ?? 10;
    const idWarn = options.identityDeviationWarning ?? 0.5;
    const idReject = options.identityDeviationReject ?? 2.0;
    const rrWarn = options.rankRatioWarning ?? 0.5;

    // 1. SVD を計算
    const numSV = options.numSingularValues || Math.min(d, 64);
    const svdResult = svd(matrix, d, d, numSV);

    // 2. 条件数
    const cn = conditionNumber(svdResult.S);
    if (cn > cnReject) {
      warnings.push(
        `条件数が非常に高い (${cn.toFixed(1)})。数値的に不安定です。`,
      );
      hasReject = true;
    } else if (cn > cnWarn) {
      warnings.push(
        `条件数がやや高い (${cn.toFixed(1)})。過度な変形の可能性があります。`,
      );
    }

    // 3. スペクトルノルム
    const sn = spectralNorm(svdResult.S);
    if (sn > snReject) {
      warnings.push(
        `スペクトルノルムが非常に高い (${sn.toFixed(3)})。入力の微小差が過度に増幅されます。`,
      );
      hasReject = true;
    } else if (sn > snWarn) {
      warnings.push(
        `スペクトルノルムがやや高い (${sn.toFixed(3)})。`,
      );
    }

    // 4. 単位行列からの乖離度
    const id = identityDeviation(matrix, d);
    if (id > idReject) {
      warnings.push(
        `単位行列からの乖離度が非常に大きい (${id.toFixed(4)})。空間構造が大きく変形されています。`,
      );
      hasReject = true;
    } else if (id > idWarn) {
      warnings.push(
        `単位行列からの乖離度がやや大きい (${id.toFixed(4)})。`,
      );
    }

    // 5. ランク比率（有効ランク = σ > σ_max * 1e-4 の数）
    const threshold = svdResult.S[0] * 1e-4;
    let effectiveRank = 0;
    for (let i = 0; i < svdResult.S.length; i++) {
      if (svdResult.S[i] > threshold) effectiveRank++;
    }
    const rr = effectiveRank / numSV;
    if (rr < rrWarn) {
      warnings.push(
        `有効ランクが低い (${effectiveRank}/${numSV}, ratio=${rr.toFixed(3)})。情報の損失が発生しています。`,
      );
    }

    // 6. 総合判定
    let verdict: "GOOD" | "WARNING" | "REJECT";
    if (hasReject) {
      verdict = "REJECT";
    } else if (warnings.length > 0) {
      verdict = "WARNING";
    } else {
      verdict = "GOOD";
    }

    return {
      conditionNumber: cn,
      spectralNorm: sn,
      identityDeviation: id,
      rankRatio: rr,
      verdict,
      warnings,
    };
  }
}
