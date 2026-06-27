import { WarpPipeline, PipelineState, FinalStageState, InputVector, OutputVector, RunContext } from "@warpvector/core";
import { WarpWorkerPool } from "./WarpWorkerPool";

export interface WarpWorkerClientOptions {
  workerScript: string | URL;
  numWorkers?: number;
  pipelineState: { steps: PipelineState[]; finalStage?: FinalStageState };
}

export class WarpWorkerClient {
  private pool: WarpWorkerPool;

  constructor(private options: WarpWorkerClientOptions) {
    this.pool = new WarpWorkerPool({
      workerScript: options.workerScript,
      numWorkers: options.numWorkers,
    });
  }

  /**
   * Initializes the pipeline on all workers.
   */
  public async init(): Promise<void> {
    await this.initWorkers(this.options.pipelineState);
  }

  // We actually need to initialize by passing the state in init, wait, we can pass state to broadcast
  public async initWorkers(pipelineState: { steps: PipelineState[]; finalStage?: FinalStageState }) {
    await this.pool.broadcast("init", pipelineState);
  }

  public async runBatch(vectors: InputVector[], context?: RunContext): Promise<OutputVector[]> {
    // Serialize input vectors properly depending on environment
    // For now we just pass them
    const result = await this.pool.executeTask("runBatch", { vectors, context });
    return result as OutputVector[];
  }

  public async terminate() {
    await this.pool.terminate();
  }
}
