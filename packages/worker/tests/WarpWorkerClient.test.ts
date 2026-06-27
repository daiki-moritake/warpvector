import { describe, it, expect, afterAll } from "bun:test";
import { WarpWorkerClient } from "../src/WarpWorkerClient";
import { WarpPipeline, InputVector } from "@warpvector/core";

describe("WarpWorkerClient", () => {
  let client: WarpWorkerClient;

  it("should initialize workers", async () => {
    client = new WarpWorkerClient({
      workerScript: new URL("./dummy-worker.ts", import.meta.url).pathname,
      numWorkers: 2,
      pipelineState: { steps: [] } // Just a dummy, we won't use it in init directly here if we use initWorkers
    });

    // We simulate a pipeline with a DummyAdapter (see dummy-worker.ts)
    const pipelineState = {
      steps: [
        { type: "DummyAdapter", state: { factor: 2 } }
      ]
    };

    await client.initWorkers(pipelineState);
    expect(true).toBe(true); // If initWorkers doesn't throw, it's successful
  });

  it("should process batches across workers", async () => {
    const batch: InputVector[] = [
      [1, 2, 3],
      [4, 5, 6]
    ];

    const results = await client.runBatch(batch);
    
    expect(results).toHaveLength(2);
    // DummyAdapter multiplies by 2
    expect(Array.from(results[0])).toEqual([2, 4, 6]);
    expect(Array.from(results[1])).toEqual([8, 10, 12]);
  });

  afterAll(async () => {
    if (client) {
      await client.terminate();
    }
  });
});
