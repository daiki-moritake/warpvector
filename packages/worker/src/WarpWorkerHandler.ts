import { WarpPipeline } from "@warpvector/core";
import { WorkerMessage, WorkerResponse } from "./types";

export class WarpWorkerHandler {
  private pipeline: WarpPipeline | null = null;

  /**
   * Must be called in the global scope of the worker script.
   * Connects the worker to the parent.
   */
  public listen() {
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
      // Node.js worker
       
      const { parentPort } = require('worker_threads');
      if (parentPort) {
        parentPort.on('message', async (msg: WorkerMessage) => {
          const response = await this.handleMessage(msg);
          parentPort.postMessage(response);
        });
      }
    } else if (typeof self !== 'undefined') {
      // Web Worker
      self.onmessage = async (e: MessageEvent) => {
        const msg = e.data as WorkerMessage;
        const response = await this.handleMessage(msg);
        self.postMessage(response);
      };
    }
  }

  private async handleMessage(msg: WorkerMessage): Promise<WorkerResponse> {
    try {
      if (msg.type === "init") {
        this.pipeline = WarpPipeline.importState(msg.payload);
        await this.pipeline.init();
        return { id: msg.id, success: true };
      }

      if (msg.type === "runBatch") {
        if (!this.pipeline) {
          throw new Error("Worker pipeline not initialized.");
        }
        const { vectors, context } = msg.payload;
        // In Web workers, Float32Arrays sent back and forth might need to be converted or transferred
        const results = await this.pipeline.runBatch(vectors, context);
        return { id: msg.id, success: true, payload: results };
      }

      throw new Error(`Unknown message type: ${msg.type}`);
    } catch (e: any) {
      return { id: msg.id, success: false, error: e.message || String(e) };
    }
  }
}
