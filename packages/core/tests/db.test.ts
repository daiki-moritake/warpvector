import { expect, test, describe } from "bun:test";
import { VectorDBAdapter } from "../src/adapters/VectorDBAdapter";

describe("VectorDBAdapter", () => {
  test("toPgvector converts a vector to a string format for PostgreSQL", () => {
    const vector = [0.1, 0.2, 0.3];
    const pgvectorString = VectorDBAdapter.toPgvector(vector);
    expect(pgvectorString).toBe("[0.1, 0.2, 0.3]");
  });

  test("toPgvector works with Float32Array", () => {
    const vector = new Float32Array([0.5, 0.6, 0.7]);
    const pgvectorString = VectorDBAdapter.toPgvector(vector);
    expect(pgvectorString).toBe("[0.5, 0.6000000238418579, 0.699999988079071]");
  });

  test("toPineconeQuery creates a query object without filters", () => {
    const vector = [0.1, 0.2, 0.3];
    const query = VectorDBAdapter.toPineconeQuery(vector, 5);
    expect(query).toEqual({
      vector: [0.1, 0.2, 0.3],
      topK: 5,
    });
  });

  test("toPineconeQuery creates a query object with filters", () => {
    const vector = [0.1, 0.2, 0.3];
    const filter = { category: "news" };
    const query = VectorDBAdapter.toPineconeQuery(vector, 10, filter);
    expect(query).toEqual({
      vector: [0.1, 0.2, 0.3],
      topK: 10,
      filter: { category: "news" },
    });
  });

  test("toRedis creates a Uint8Array binary format of Float32Array", () => {
    const vector = [1.0, 2.0, 3.0];
    const redisBinary = VectorDBAdapter.toRedis(vector);
    expect(redisBinary instanceof Uint8Array).toBe(true);

    // Float32のバイナリが正しく生成されているか確認する
    const floatView = new Float32Array(
      redisBinary.buffer,
      redisBinary.byteOffset,
      redisBinary.byteLength / 4,
    );
    expect(floatView[0]).toBe(1.0);
    expect(floatView[1]).toBe(2.0);
    expect(floatView[2]).toBe(3.0);
  });

  test("rejects invalid vectors with NaN, Infinity, or incorrect types", () => {
    const nanVector = [1.0, NaN, 3.0];
    const infVector = [1.0, Infinity, 3.0];
    const invalidType = { values: [1.0, 2.0] } as any;

    // toPgvector
    expect(() => VectorDBAdapter.toPgvector(nanVector)).toThrow("NaN");
    expect(() => VectorDBAdapter.toPgvector(infVector)).toThrow("Infinity");
    expect(() => VectorDBAdapter.toPgvector(invalidType)).toThrow(
      "Invalid vector type",
    );

    // toPineconeQuery
    expect(() => VectorDBAdapter.toPineconeQuery(nanVector)).toThrow("NaN");
    expect(() => VectorDBAdapter.toPineconeQuery(infVector)).toThrow(
      "Infinity",
    );

    // toRedis
    expect(() => VectorDBAdapter.toRedis(nanVector)).toThrow("NaN");
    expect(() => VectorDBAdapter.toRedis(infVector)).toThrow("Infinity");

    // toVectorizeQuery
    expect(() => VectorDBAdapter.toVectorizeQuery(nanVector)).toThrow("NaN");
    expect(() => VectorDBAdapter.toVectorizeQuery(infVector)).toThrow(
      "Infinity",
    );

    // toVectorizeRecord
    expect(() =>
      VectorDBAdapter.toVectorizeRecord("doc-1", nanVector),
    ).toThrow("NaN");
    expect(() =>
      VectorDBAdapter.toVectorizeRecord("doc-1", infVector),
    ).toThrow("Infinity");
  });
});
