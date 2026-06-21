/**
 * デシリアライゼーション用の軽量バリデーションユーティリティ。
 * ゼロ依存ポリシーを維持するため、手書きのバリデーションを提供します。
 * importState 等で外部から渡される JSON データの安全な復元に使用します。
 */

import { WarpValidationError } from "./errors";

/**
 * 値が指定された型であることを検証します。
 *
 * @param value 検証する値
 * @param type 期待する typeof の結果 (例: "string", "number")
 * @param field フィールド名（エラーメッセージに表示）
 * @param component コンポーネント名（省略時は "WarpVector"）
 * @throws WarpValidationError 型が一致しない場合
 */
export function assertType(
  value: unknown,
  type: string,
  field: string,
  component?: string,
): void {
  if (typeof value !== type) {
    throw new WarpValidationError(
      component || "WarpVector",
      field,
      `${type} 型の値が必要ですが、${typeof value} 型が渡されました。`,
    );
  }
}

/**
 * 値が正の整数であることを検証します。
 *
 * @returns 検証済みの値
 * @throws WarpValidationError 正の整数でない場合
 */
export function assertPositiveInt(
  value: unknown,
  field: string,
  component?: string,
): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new WarpValidationError(
      component || "WarpVector",
      field,
      `正の整数が必要ですが、${JSON.stringify(value)} が渡されました。`,
    );
  }
  return value;
}

/**
 * 値が非負の整数であることを検証します。
 *
 * @returns 検証済みの値
 * @throws WarpValidationError 非負の整数でない場合
 */
export function assertNonNegativeInt(
  value: unknown,
  field: string,
  component?: string,
): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new WarpValidationError(
      component || "WarpVector",
      field,
      `非負の整数が必要ですが、${JSON.stringify(value)} が渡されました。`,
    );
  }
  return value;
}

/**
 * 値が配列であることを検証します。
 *
 * @returns 検証済みの配列
 * @throws WarpValidationError 配列でない場合
 */
export function assertArray(
  value: unknown,
  field: string,
  component?: string,
): unknown[] {
  if (!Array.isArray(value)) {
    throw new WarpValidationError(
      component || "WarpVector",
      field,
      `配列が必要ですが、${typeof value} 型が渡されました。`,
    );
  }
  return value;
}

/**
 * 値が有限な数値の配列であることを検証します。
 *
 * @returns 検証済みの数値配列
 * @throws WarpValidationError 数値配列でない、またはNaN/Infinityを含む場合
 */
export function assertNumberArray(
  value: unknown,
  field: string,
  component?: string,
): number[] {
  const arr = assertArray(value, field, component);
  for (let i = 0; i < arr.length; i++) {
    if (typeof arr[i] !== "number" || !Number.isFinite(arr[i] as number)) {
      throw new WarpValidationError(
        component || "WarpVector",
        field,
        `数値配列の ${field}[${i}] が有限な数値ではありません（値: ${arr[i]}）。NaN や Infinity は使用できません。`,
      );
    }
  }
  return arr as number[];
}

/**
 * 値がオブジェクト（null でない）であることを検証します。
 *
 * @returns 検証済みのオブジェクト
 * @throws WarpValidationError オブジェクトでない、またはnullの場合
 */
export function assertObject(
  value: unknown,
  field: string,
  component?: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new WarpValidationError(
      component || "WarpVector",
      field,
      `非nullオブジェクトが必要ですが、${value === null ? "null" : typeof value} が渡されました。`,
    );
  }
  return value as Record<string, unknown>;
}

/**
 * ベクトルの次元数を検証します。
 *
 * @param vector 検証するベクトル
 * @param expectedDim 期待される次元数
 * @param component コンポーネント名（エラーメッセージに表示）
 * @throws WarpValidationError 次元が一致しない場合
 *
 * @example
 * ```typescript
 * assertVectorDim(inputVector, 1536, "IntentAdapter");
 * // → 次元が一致しない場合:
 * // "IntentAdapter: フィールド 'vector' のバリデーションに失敗しました。
 * //   期待された次元数: 1536, 実際の次元数: 768"
 * ```
 */
export function assertVectorDim(
  vector: number[] | Float32Array,
  expectedDim: number,
  component: string,
): void {
  if (vector.length !== expectedDim) {
    throw new WarpValidationError(
      component,
      "vector",
      `期待された次元数: ${expectedDim}, 実際の次元数: ${vector.length}`,
    );
  }
}

/**
 * JSON 文字列を安全にパースします。
 *
 * @returns パース結果
 * @throws WarpValidationError パースに失敗した場合、コンテキスト付きのエラーメッセージをスロー
 */
export function safeJsonParse(json: string, context: string): unknown {
  if (typeof json !== "string") {
    throw new WarpValidationError(
      context,
      "state",
      `JSON文字列が必要ですが、${typeof json} 型が渡されました。importState() に渡すデータが exportState() の出力であることを確認してください。`,
    );
  }
  try {
    return JSON.parse(json);
  } catch (e) {
    throw new WarpValidationError(
      context,
      "state",
      `JSONのパースに失敗しました: ${(e as Error).message}`,
    );
  }
}
