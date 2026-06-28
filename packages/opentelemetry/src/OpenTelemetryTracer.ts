import {
  trace,
  context,
  Tracer as OTelTracer,
  SpanStatusCode,
} from "@opentelemetry/api";
import { WarpTracer, SpanAttributes, WarpMetrics } from "@warpvector/core";

export interface OpenTelemetryTracerOptions {
  name: string;
  version?: string;
}

export class OpenTelemetryTracer extends WarpTracer {
  private tracer: OTelTracer;

  constructor(options: OpenTelemetryTracerOptions = { name: "warpvector" }) {
    super();
    this.tracer = trace.getTracer(options.name, options.version);
  }

  /**
   * Overrides the trace method to create an OpenTelemetry span.
   */
  trace<T>(operationName: string, attributes: SpanAttributes, fn: () => T): T {
    return this.tracer.startActiveSpan(operationName, (span) => {
      span.setAttributes(attributes);

      try {
        const result = super.trace(operationName, attributes, fn);
        span.setStatus({ code: SpanStatusCode.OK });
        return result;
      } catch (err: any) {
        span.recordException(err);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: err.message,
        });
        throw err;
      } finally {
        span.end();
      }
    });
  }

  /**
   * Overrides the traceAsync method to create an OpenTelemetry span.
   */
  async traceAsync<T>(
    operationName: string,
    attributes: SpanAttributes,
    fn: () => Promise<T>,
  ): Promise<T> {
    return this.tracer.startActiveSpan(operationName, async (span) => {
      span.setAttributes(attributes);

      try {
        const result = await super.traceAsync(operationName, attributes, fn);
        span.setStatus({ code: SpanStatusCode.OK });
        return result;
      } catch (err: any) {
        span.recordException(err);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: err.message,
        });
        throw err;
      } finally {
        span.end();
      }
    });
  }
}
