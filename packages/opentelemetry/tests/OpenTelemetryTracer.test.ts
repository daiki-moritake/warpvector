import { describe, it, expect } from "bun:test";
import { OpenTelemetryTracer } from "../src/OpenTelemetryTracer";
import { trace, context } from "@opentelemetry/api";

describe("OpenTelemetryTracer", () => {
  it("should record internal metrics and create spans for trace()", () => {
    const tracer = new OpenTelemetryTracer();
    const result = tracer.trace("test-sync-op", { foo: "bar" }, () => {
      return 42;
    });

    expect(result).toBe(42);

    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.operationCounts["test-sync-op"]).toBe(1);
  });

  it("should record internal metrics and create spans for traceAsync()", async () => {
    const tracer = new OpenTelemetryTracer();
    const result = await tracer.traceAsync(
      "test-async-op",
      { test: true },
      async () => {
        return "async-result";
      },
    );

    expect(result).toBe("async-result");

    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.operationCounts["test-async-op"]).toBe(1);
  });

  it("should throw errors and record them correctly in trace()", () => {
    const tracer = new OpenTelemetryTracer();

    let error = null;
    try {
      tracer.trace("test-sync-error", {}, () => {
        throw new Error("Sync Error");
      });
    } catch (e) {
      error = e;
    }

    expect(error).not.toBeNull();

    const metrics = tracer.getMetrics();
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.operationCounts["test-sync-error"]).toBe(1);
  });
});
