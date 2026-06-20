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

describe("validation utilities", () => {
  // --- safeJsonParse ---
  describe("safeJsonParse", () => {
    test("parses valid JSON", () => {
      expect(safeJsonParse('{"a": 1}', "test")).toEqual({ a: 1 });
    });

    test("throws on invalid JSON with context", () => {
      expect(() => safeJsonParse("{bad}", "TestAdapter")).toThrow(
        "Failed to parse JSON for TestAdapter",
      );
    });

    test("throws on non-string input", () => {
      expect(() => safeJsonParse(42 as any, "TestAdapter")).toThrow(
        "expected a JSON string",
      );
    });

    test("throws on null input", () => {
      expect(() => safeJsonParse(null as any, "TestAdapter")).toThrow(
        "expected a JSON string",
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
        "field 'myField' must be string, got number",
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
      expect(() => assertPositiveInt(0, "dim")).toThrow("positive integer");
    });

    test("throws for negative", () => {
      expect(() => assertPositiveInt(-1, "dim")).toThrow("positive integer");
    });

    test("throws for float", () => {
      expect(() => assertPositiveInt(1.5, "dim")).toThrow("positive integer");
    });

    test("throws for string", () => {
      expect(() => assertPositiveInt("3" as any, "dim")).toThrow(
        "positive integer",
      );
    });

    test("throws for NaN", () => {
      expect(() => assertPositiveInt(NaN, "dim")).toThrow("positive integer");
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
      expect(() => assertNonNegativeInt(-1, "count")).toThrow("non-negative");
    });
  });

  // --- assertArray ---
  describe("assertArray", () => {
    test("passes for arrays", () => {
      expect(assertArray([], "arr")).toEqual([]);
      expect(assertArray([1, 2], "arr")).toEqual([1, 2]);
    });

    test("throws for non-array", () => {
      expect(() => assertArray({}, "arr")).toThrow("must be an array");
      expect(() => assertArray("str", "arr")).toThrow("must be an array");
      expect(() => assertArray(null, "arr")).toThrow("must be an array");
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
        "finite number",
      );
    });

    test("throws for array with Infinity", () => {
      expect(() => assertNumberArray([1, Infinity], "vec")).toThrow(
        "finite number",
      );
    });

    test("throws for array with string", () => {
      expect(() => assertNumberArray([1, "two" as any, 3], "vec")).toThrow(
        "finite number",
      );
    });

    test("throws for non-array", () => {
      expect(() => assertNumberArray("not_array" as any, "vec")).toThrow(
        "must be an array",
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
      expect(() => assertObject(null, "obj")).toThrow("non-null object");
    });

    test("throws for array", () => {
      expect(() => assertObject([], "obj")).toThrow("non-null object");
    });

    test("throws for string", () => {
      expect(() => assertObject("str", "obj")).toThrow("non-null object");
    });
  });
});
