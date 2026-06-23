/**
 * WarpTracer テスト — OpenTelemetry 互換計装のテスト
 */
import { describe, test, expect } from "bun:test";
import { WarpTracer } from "@warpvector/core";

describe("WarpTracer", () => {
  test("trace() が関数の戻り値を透過的に返す", () => {
    const tracer = new WarpTracer();
    const result = tracer.trace("test-op", {}, () => 42);
    expect(result).toBe(42);
  });

  test("trace() がレイテンシを正しく記録する", () => {
    const tracer = new WarpTracer();
    tracer.trace("op-a", { dim: 768 }, () => {
      // 意図的にビジーウェイト
      let sum = 0;
      for (let i = 0; i < 100_000; i++) sum += i;
      return sum;
    });

    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.avgLatencyMs).toBeGreaterThan(0);
    expect(metrics.operationCounts["op-a"]).toBe(1);
  });

  test("複数の操作が個別に記録される", () => {
    const tracer = new WarpTracer();
    tracer.trace("tune", { intent: "tech" }, () => "a");
    tracer.trace("tune", { intent: "biz" }, () => "b");
    tracer.trace("run", {}, () => "c");

    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(3);
    expect(metrics.operationCounts["tune"]).toBe(2);
    expect(metrics.operationCounts["run"]).toBe(1);
  });

  test("traceAsync() が非同期関数を正しくトレースする", async () => {
    const tracer = new WarpTracer();
    const result = await tracer.traceAsync("async-op", {}, async () => {
      return "async-result";
    });

    expect(result).toBe("async-result");
    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.operationCounts["async-op"]).toBe(1);
  });

  test("エラー発生時もメトリクスが記録され、エラーが再スローされる", () => {
    const tracer = new WarpTracer();
    expect(() =>
      tracer.trace("failing-op", {}, () => {
        throw new Error("intentional error");
      }),
    ).toThrow("intentional error");

    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.operationCounts["failing-op"]).toBe(1);
  });

  test("resetMetrics() がメトリクスを完全にクリアする", () => {
    const tracer = new WarpTracer();
    tracer.trace("a", {}, () => 1);
    tracer.trace("b", {}, () => 2);

    const before = tracer.getMetrics();
    expect(before.totalCalls).toBe(2);

    tracer.resetMetrics();

    const after = tracer.getMetrics();
    expect(after.totalCalls).toBe(0);
    expect(after.avgLatencyMs).toBe(0);
    expect(after.maxLatencyMs).toBe(0);
    expect(after.minLatencyMs).toBe(0);
    expect(Object.keys(after.operationCounts)).toHaveLength(0);
  });

  test("min/max レイテンシが正しく追跡される", () => {
    const tracer = new WarpTracer();
    // 軽い処理
    tracer.trace("fast", {}, () => 1);
    // 少し重い処理
    tracer.trace("slow", {}, () => {
      let sum = 0;
      for (let i = 0; i < 500_000; i++) sum += i;
      return sum;
    });

    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(2);
    expect(metrics.maxLatencyMs).toBeGreaterThanOrEqual(metrics.minLatencyMs);
  });

  test("空のメトリクスが安全なデフォルト値を返す", () => {
    const tracer = new WarpTracer();
    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(0);
    expect(metrics.avgLatencyMs).toBe(0);
    expect(metrics.maxLatencyMs).toBe(0);
    expect(metrics.minLatencyMs).toBe(0);
    expect(Object.keys(metrics.operationCounts)).toHaveLength(0);
  });
});

describe("VectorDBAdapter Vectorize", () => {
  test("toVectorizeQuery() が正しい形式のクエリを返す", async () => {
    const { VectorDBAdapter } = await import("@warpvector/core");
    const vec = new Float32Array([0.1, 0.2, 0.3]);
    const query = VectorDBAdapter.toVectorizeQuery(vec, 5, {
      returnMetadata: true,
    });

    expect(query.vector).toHaveLength(3);
    expect(query.vector[0]).toBeCloseTo(0.1, 5);
    expect(query.options.topK).toBe(5);
    expect(query.options.returnMetadata).toBe(true);
  });

  test("toVectorizeQuery() がデフォルト topK=10 を使用", async () => {
    const { VectorDBAdapter } = await import("@warpvector/core");
    const query = VectorDBAdapter.toVectorizeQuery([1, 2, 3]);
    expect(query.options.topK).toBe(10);
  });

  test("toVectorizeRecord() が正しい形式のレコードを返す", async () => {
    const { VectorDBAdapter } = await import("@warpvector/core");
    const record = VectorDBAdapter.toVectorizeRecord(
      "doc-1",
      new Float32Array([0.5, 0.6]),
      { title: "test" },
    );

    expect(record.id).toBe("doc-1");
    expect(record.values).toHaveLength(2);
    expect(record.values[0]).toBeCloseTo(0.5, 5);
    expect(record.metadata).toEqual({ title: "test" });
  });

  test("toVectorizeRecord() がメタデータなしで動作", async () => {
    const { VectorDBAdapter } = await import("@warpvector/core");
    const record = VectorDBAdapter.toVectorizeRecord("doc-2", [1, 2]);
    expect(record.id).toBe("doc-2");
    expect(record.values).toEqual([1, 2]);
    expect(record.metadata).toBeUndefined();
  });
});
