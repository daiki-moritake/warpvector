/**
 * WarpVector 構造化エラーモジュール
 *
 * パイプラインのどのステップで、なぜ失敗したかを明確にするための
 * 構造化エラークラス群を提供します。
 */

/**
 * WarpVector のすべてのエラーの基底クラス。
 * `instanceof WarpError` で WarpVector 由来のエラーかどうかを判定できます。
 */
export class WarpError extends Error {
  /** エラーコード。プログラム上での分岐に使用します。 */
  public readonly code: string;

  constructor(code: string, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "WarpError";
    this.code = code;
  }
}

/**
 * パイプラインの特定ステップで発生したエラー。
 *
 * @example
 * ```typescript
 * try {
 *   pipeline.run(vector, { intent: "tech" });
 * } catch (e) {
 *   if (e instanceof WarpPipelineError) {
 *     console.error(`Step ${e.stepIndex} (${e.stepType}) failed: ${e.message}`);
 *   }
 * }
 * ```
 */
export class WarpPipelineError extends WarpError {
  /** エラーが発生したパイプラインステップのインデックス（0始まり） */
  public readonly stepIndex: number;
  /** エラーが発生したパイプラインステップの型名（例: "MlpAdapter"） */
  public readonly stepType: string;

  constructor(
    message: string,
    stepIndex: number,
    stepType: string,
    options?: ErrorOptions,
  ) {
    super(
      "PIPELINE_STEP_ERROR",
      `WarpPipeline [Step ${stepIndex}: ${stepType}] ${message}`,
      options,
    );
    this.name = "WarpPipelineError";
    this.stepIndex = stepIndex;
    this.stepType = stepType;
  }
}

/**
 * ベクトルの次元が期待と一致しないときにスローされるエラー。
 *
 * @example
 * ```typescript
 * // "IntentAdapter: 入力ベクトルの次元が一致しません。期待: 1536, 実際: 768"
 * throw new WarpDimensionMismatchError("IntentAdapter", 1536, 768);
 * ```
 */
export class WarpDimensionMismatchError extends WarpError {
  /** 期待された次元数 */
  public readonly expectedDim: number;
  /** 実際に渡された次元数 */
  public readonly actualDim: number;
  /** エラーが発生したコンポーネント名 */
  public readonly component: string;

  constructor(
    component: string,
    expectedDim: number,
    actualDim: number,
    hint?: string,
  ) {
    const hintMsg = hint ? `\n  ヒント: ${hint}` : "";
    super(
      "DIMENSION_MISMATCH",
      `${component}: 入力ベクトルの次元が一致しません。\n` +
        `  期待: ${expectedDim}\n` +
        `  実際: ${actualDim}${hintMsg}`,
    );
    this.name = "WarpDimensionMismatchError";
    this.expectedDim = expectedDim;
    this.actualDim = actualDim;
    this.component = component;
  }
}

/**
 * WASMまたはアダプタの初期化が完了していない状態でメソッドを呼んだときのエラー。
 *
 * @example
 * ```typescript
 * throw new WarpInitializationError(
 *   "MlpAdapter",
 *   "WASMが初期化されていません。await pipeline.init() を実行してください。"
 * );
 * ```
 */
export class WarpInitializationError extends WarpError {
  /** エラーが発生したコンポーネント名 */
  public readonly component: string;

  constructor(component: string, message?: string) {
    super(
      "NOT_INITIALIZED",
      `${component}: ${message || "初期化が完了していません。await pipeline.init() を実行してください。"}`,
    );
    this.name = "WarpInitializationError";
    this.component = component;
  }
}

/**
 * importState や設定オブジェクトのバリデーションに失敗したときのエラー。
 *
 * @example
 * ```typescript
 * throw new WarpValidationError(
 *   "IntentAdapter",
 *   "matrix",
 *   "2次元の数値配列（正方行列）が必要です。"
 * );
 * ```
 */
export class WarpValidationError extends WarpError {
  /** エラーが発生したコンポーネント名 */
  public readonly component: string;
  /** 問題のあるフィールド名 */
  public readonly field: string;

  constructor(component: string, field: string, detail: string) {
    super(
      "VALIDATION_ERROR",
      `${component}: フィールド '${field}' のバリデーションに失敗しました。\n  ${detail}`,
    );
    this.name = "WarpValidationError";
    this.component = component;
    this.field = field;
  }
}
