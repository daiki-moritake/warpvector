/**
 * デシリアライゼーション用の軽量バリデーションユーティリティ。
 * ゼロ依存ポリシーを維持するため、手書きのバリデーションを提供します。
 * importState 等で外部から渡される JSON データの安全な復元に使用します。
 */

/**
 * 値が指定された型であることを検証します。
 * @throws 型が一致しない場合
 */
export function assertType(value: unknown, type: string, field: string): void {
  if (typeof value !== type) {
    throw new Error(
      `Invalid state: field '${field}' must be ${type}, got ${typeof value}`,
    );
  }
}

/**
 * 値が正の整数であることを検証します。
 * @returns 検証済みの値
 * @throws 正の整数でない場合
 */
export function assertPositiveInt(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(
      `Invalid state: field '${field}' must be a positive integer, got ${value}`,
    );
  }
  return value;
}

/**
 * 値が非負の整数であることを検証します。
 * @returns 検証済みの値
 * @throws 非負の整数でない場合
 */
export function assertNonNegativeInt(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error(
      `Invalid state: field '${field}' must be a non-negative integer, got ${value}`,
    );
  }
  return value;
}

/**
 * 値が配列であることを検証します。
 * @returns 検証済みの配列
 * @throws 配列でない場合
 */
export function assertArray(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`Invalid state: field '${field}' must be an array`);
  }
  return value;
}

/**
 * 値が有限な数値の配列であることを検証します。
 * @returns 検証済みの数値配列
 * @throws 数値配列でない、またはNaN/Infinityを含む場合
 */
export function assertNumberArray(value: unknown, field: string): number[] {
  const arr = assertArray(value, field);
  for (let i = 0; i < arr.length; i++) {
    if (typeof arr[i] !== "number" || !isFinite(arr[i] as number)) {
      throw new Error(
        `Invalid state: field '${field}[${i}]' must be a finite number, got ${arr[i]}`,
      );
    }
  }
  return arr as number[];
}

/**
 * 値がオブジェクト（null でない）であることを検証します。
 * @returns 検証済みのオブジェクト
 * @throws オブジェクトでない、またはnullの場合
 */
export function assertObject(
  value: unknown,
  field: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(
      `Invalid state: field '${field}' must be a non-null object`,
    );
  }
  return value as Record<string, unknown>;
}

/**
 * JSON 文字列を安全にパースします。
 * @returns パース結果
 * @throws パースに失敗した場合、コンテキスト付きのエラーメッセージをスロー
 */
export function safeJsonParse(json: string, context: string): unknown {
  if (typeof json !== "string") {
    throw new Error(
      `Failed to parse state for ${context}: expected a JSON string, got ${typeof json}`,
    );
  }
  try {
    return JSON.parse(json);
  } catch (e) {
    throw new Error(
      `Failed to parse JSON for ${context}: ${(e as Error).message}`,
    );
  }
}
