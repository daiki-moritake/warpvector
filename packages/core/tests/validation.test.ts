import { describe, expect, test } from "bun:test";
import {
  safeJsonParse,
  assertType,
  assertPositiveInt,
  assertNonNegativeInt,
  assertArray,
  assertNumberArray,
  assertObject,
} from "../src/validation";
import { WarpValidationError } from "../src/errors";

describe("validation utilities", () => {
  // --- safeJsonParse ---
  describe("safeJsonParse", () => {
    test("parses valid JSON", () => {
      expect(safeJsonParse('{"a": 1}', "test")).toEqual({ a: 1 });
    });

    test("throws on invalid JSON with context", () => {
      expect(() => safeJsonParse("{bad}", "TestAdapter")).toThrow(
        WarpValidationError,
      );
      expect(() => safeJsonParse("{bad}", "TestAdapter")).toThrow(
        "JSONのパースに失敗しました",
      );
    });

    test("throws on non-string input", () => {
      expect(() => safeJsonParse(42 as any, "TestAdapter")).toThrow(
        WarpValidationError,
      );
      expect(() => safeJsonParse(42 as any, "TestAdapter")).toThrow(
        "JSON文字列が必要です",
      );
    });

    test("throws on null input", () => {
      expect(() => safeJsonParse(null as any, "TestAdapter")).toThrow(
        WarpValidationError,
      );
    });
  });

  // --- assertType ---
  describe("assertType", () => {
    test("passes for correct type", () => {
      expect(() => assertType("hello", "string", "field")).not.toThrow();
      expect(() => assertType(42, "number", "field")).not.toThrow();
      expect(() => assertType(true, "boolean", "field")).not.toThrow();
    });

    test("throws for wrong type", () => {
      expect(() => assertType(42, "string", "myField")).toThrow(
        WarpValidationError,
      );
      expect(() => assertType(42, "string", "myField")).toThrow(
        "string",
      );
    });
  });

  // --- assertPositiveInt ---
  describe("assertPositiveInt", () => {
    test("passes for positive integers", () => {
      expect(assertPositiveInt(1, "dim")).toBe(1);
      expect(assertPositiveInt(100, "dim")).toBe(100);
    });

    test("throws for zero", () => {
      expect(() => assertPositiveInt(0, "dim")).toThrow("正の整数");
    });

    test("throws for negative", () => {
      expect(() => assertPositiveInt(-1, "dim")).toThrow("正の整数");
    });

    test("throws for float", () => {
      expect(() => assertPositiveInt(1.5, "dim")).toThrow("正の整数");
    });

    test("throws for string", () => {
      expect(() => assertPositiveInt("3" as any, "dim")).toThrow(
        "正の整数",
      );
    });

    test("throws for NaN", () => {
      expect(() => assertPositiveInt(NaN, "dim")).toThrow("正の整数");
    });
  });

  // --- assertNonNegativeInt ---
  describe("assertNonNegativeInt", () => {
    test("passes for zero", () => {
      expect(assertNonNegativeInt(0, "count")).toBe(0);
    });

    test("passes for positive", () => {
      expect(assertNonNegativeInt(5, "count")).toBe(5);
    });

    test("throws for negative", () => {
      expect(() => assertNonNegativeInt(-1, "count")).toThrow("非負の整数");
    });
  });

  // --- assertArray ---
  describe("assertArray", () => {
    test("passes for arrays", () => {
      expect(assertArray([], "arr")).toEqual([]);
      expect(assertArray([1, 2], "arr")).toEqual([1, 2]);
    });

    test("throws for non-array", () => {
      expect(() => assertArray({}, "arr")).toThrow("配列が必要です");
      expect(() => assertArray("str", "arr")).toThrow("配列が必要です");
      expect(() => assertArray(null, "arr")).toThrow("配列が必要です");
    });
  });

  // --- assertNumberArray ---
  describe("assertNumberArray", () => {
    test("passes for valid number arrays", () => {
      expect(assertNumberArray([1, 2, 3], "vec")).toEqual([1, 2, 3]);
      expect(assertNumberArray([0, -1.5, 3.14], "vec")).toEqual([
        0, -1.5, 3.14,
      ]);
    });

    test("throws for array with NaN", () => {
      expect(() => assertNumberArray([1, NaN, 3], "vec")).toThrow(
        "有限な数値ではありません",
      );
    });

    test("throws for array with Infinity", () => {
      expect(() => assertNumberArray([1, Infinity], "vec")).toThrow(
        "有限な数値ではありません",
      );
    });

    test("throws for array with string", () => {
      expect(() => assertNumberArray([1, "two" as any, 3], "vec")).toThrow(
        "有限な数値ではありません",
      );
    });

    test("throws for non-array", () => {
      expect(() => assertNumberArray("not_array" as any, "vec")).toThrow(
        "配列が必要です",
      );
    });
  });

  // --- assertObject ---
  describe("assertObject", () => {
    test("passes for plain objects", () => {
      expect(assertObject({}, "obj")).toEqual({});
      expect(assertObject({ key: "val" }, "obj")).toEqual({ key: "val" });
    });

    test("throws for null", () => {
      expect(() => assertObject(null, "obj")).toThrow("非nullオブジェクト");
    });

    test("throws for array", () => {
      expect(() => assertObject([], "obj")).toThrow("非nullオブジェクト");
    });

    test("throws for string", () => {
      expect(() => assertObject("str", "obj")).toThrow("非nullオブジェクト");
    });
  });

  // --- WarpValidationError properties ---
  describe("WarpValidationError properties", () => {
    test("includes component and field", () => {
      try {
        assertType(42, "string", "myField", "TestComponent");
      } catch (e) {
        expect(e).toBeInstanceOf(WarpValidationError);
        const err = e as WarpValidationError;
        expect(err.component).toBe("TestComponent");
        expect(err.field).toBe("myField");
        expect(err.code).toBe("VALIDATION_ERROR");
      }
    });
  });
});
