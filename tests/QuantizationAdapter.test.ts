import { describe, expect, test } from "bun:test";
import { QuantizationAdapter } from "../src/QuantizationAdapter";
import { normalize } from "../src/utils";

describe("QuantizationAdapter", () => {
  test("performs int8 scalar quantization correctly", () => {
    const adapter = new QuantizationAdapter({ type: "int8", dim: 4 });
    const vector = new Float32Array([1.0, -1.0, 0.5, -0.1]);
    
    const result = adapter.tune(vector) as Int8Array;
    
    expect(result).toBeInstanceOf(Int8Array);
    expect(result.length).toBe(4);
    expect(result[0]).toBe(127);
    expect(result[1]).toBe(-127);
    expect(result[2]).toBe(64); // 0.5 * 127 = 63.5 -> round to 64
    expect(result[3]).toBe(-13); // -0.1 * 127 = -12.7 -> round to -13
  });

  test("clips int8 values correctly", () => {
    const adapter = new QuantizationAdapter({ type: "int8", dim: 2 });
    const vector = new Float32Array([2.0, -3.0]);
    
    const result = adapter.tune(vector) as Int8Array;
    expect(result[0]).toBe(127); // clipped
    expect(result[1]).toBe(-128); // clipped
  });

  test("performs binary quantization and bit packing correctly", () => {
    const adapter = new QuantizationAdapter({ type: "binary", dim: 8 });
    // >0 => 1, <=0 => 0
    // [1.0, -1.0, 0.5, -0.1, 0.0, 2.0, -3.0, 0.1]
    // Bits: 1, 0, 1, 0, 0, 1, 0, 1 => 10100101 in binary => 165 in decimal
    const vector = new Float32Array([1.0, -1.0, 0.5, -0.1, 0.0, 2.0, -3.0, 0.1]);
    
    const result = adapter.tune(vector) as Uint8Array;
    
    expect(result).toBeInstanceOf(Uint8Array);
    expect(result.length).toBe(1); // 8 dim / 8 = 1 byte
    expect(result[0]).toBe(165); // 10100101
  });

  test("throws error if binary quantization dimension is not multiple of 8", () => {
    expect(() => {
      new QuantizationAdapter({ type: "binary", dim: 10 });
    }).toThrow();
  });

  test("computes int8 dot product correctly", () => {
    const a = new Int8Array([10, -5, 2]);
    const b = new Int8Array([3, 4, -1]);
    // 10*3 + (-5)*4 + 2*(-1) = 30 - 20 - 2 = 8
    const dot = QuantizationAdapter.int8DotProduct(a, b);
    expect(dot).toBe(8);
  });

  test("computes hamming distance correctly", () => {
    // 10100101 (165)
    // 11110000 (240)
    // XOR: 01010101 => 4 bits differ
    const a = new Uint8Array([165]);
    const b = new Uint8Array([240]);
    
    const distance = QuantizationAdapter.hammingDistance(a, b);
    expect(distance).toBe(4);
  });
});
