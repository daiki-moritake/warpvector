import { describe, it, expect, afterAll } from "bun:test";
import { WarpWorkerClient } from "../src/WarpWorkerClient";
import { WarpPipeline, InputVector } from "@warpvector/core";

describe("WarpWorkerClient", () => {
  let client: WarpWorkerClient;

  it("should initialize workers", async () => {
    client = new WarpWorkerClient({
      workerScript: new URL("./dummy-worker.ts", import.meta.url).pathname,
      numWorkers: 2,
      pipelineState: { steps: [] }, // Just a dummy, we won't use it in init directly here if we use initWorkers
    });

    // We simulate a pipeline with a DummyAdapter (see dummy-worker.ts)
    const pipelineState = {
      steps: [{ type: "DummyAdapter", state: { factor: 2 } }],
    };

    await client.initWorkers(pipelineState);
    expect(true).toBe(true); // If initWorkers doesn't throw, it's successful
  });

  it("should process batches across workers", async () => {
    const batch: InputVector[] = [
      [1, 2, 3],
      [4, 5, 6],
    ];

    const results = await client.runBatch(batch);

    expect(results).toHaveLength(2);
    // DummyAdapter multiplies by 2
    expect(Array.from(results[0])).toEqual([2, 4, 6]);
    expect(Array.from(results[1])).toEqual([8, 10, 12]);
  });

  it("should broadcast safely without duplicating idle workers", async () => {
    const pool = client["pool"];
    // idleWorkers の初期数は 2
    expect(pool["idleWorkers"]).toHaveLength(2);

    // ブロードキャストを送信
    await pool.broadcast("init", {
      steps: [{ type: "DummyAdapter", state: { factor: 2 } }],
    });

    // 完了後、idleWorkers の数は依然として 2 のはず (重複して増えない)
    expect(pool["idleWorkers"]).toHaveLength(2);

    // 重複したインスタンスが存在しないことの確認
    const uniqueWorkers = new Set(pool["idleWorkers"]);
    expect(uniqueWorkers.size).toBe(pool["idleWorkers"].length);
  });

  afterAll(async () => {
    if (client) {
      await client.terminate();
    }
  });
});
