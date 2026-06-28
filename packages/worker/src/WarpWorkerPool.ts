import { IsomorphicWorker, createWorker } from "./IsomorphicWorker";
import { WorkerMessage, WorkerResponse, Job } from "./types";

export interface WarpWorkerPoolOptions {
  workerScript: string | URL;
  numWorkers?: number;
}

export class WarpWorkerPool {
  private workers: IsomorphicWorker[] = [];
  private idleWorkers: IsomorphicWorker[] = [];
  private jobQueue: { workerMessage: WorkerMessage; job: Job }[] = [];
  private activeJobs: Map<number, Job> = new Map();
  private workerActiveJobs: Map<IsomorphicWorker, number> = new Map();
  private nextMessageId = 1;

  constructor(private options: WarpWorkerPoolOptions) {
    const numWorkers =
      options.numWorkers ||
      (typeof navigator !== "undefined"
        ? navigator.hardwareConcurrency || 4
        : 4);
    for (let i = 0; i < numWorkers; i++) {
      this.addWorker();
    }
  }

  private addWorker() {
    const worker = createWorker(this.options.workerScript);
    worker.onMessage((response: WorkerResponse) => {
      this.handleResponse(worker, response);
    });
    worker.onError((error) => {
      console.error("Worker error:", error);
      // Depending on the use case, you might want to terminate and replace the worker.
    });
    this.workers.push(worker);
    this.workerActiveJobs.set(worker, 0);
    this.idleWorkers.push(worker);
  }

  private handleResponse(worker: IsomorphicWorker, response: WorkerResponse) {
    if (!this.workers.includes(worker)) return;

    const job = this.activeJobs.get(response.id);
    if (job) {
      if (response.success) {
        job.resolve(response.payload);
      } else {
        job.reject(new Error(response.error));
      }
      this.activeJobs.delete(response.id);
    }

    let count = this.workerActiveJobs.get(worker) || 1;
    count--;
    this.workerActiveJobs.set(worker, count);

    // Worker is now fully idle, assign next job if any
    if (count === 0) {
      if (this.jobQueue.length > 0) {
        const nextTask = this.jobQueue.shift()!;
        this.assignJob(worker, nextTask.workerMessage, nextTask.job);
      } else {
        this.idleWorkers.push(worker);
      }
    }
  }

  private assignJob(
    worker: IsomorphicWorker,
    message: WorkerMessage,
    job: Job,
  ) {
    this.activeJobs.set(message.id, job);
    const count = this.workerActiveJobs.get(worker) || 0;
    this.workerActiveJobs.set(worker, count + 1);
    worker.postMessage(message);
  }

  public async executeTask(
    type: WorkerMessage["type"],
    payload?: any,
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = this.nextMessageId++;
      const message: WorkerMessage = { id, type, payload };
      const job: Job = { id, resolve, reject };

      if (this.idleWorkers.length > 0) {
        const worker = this.idleWorkers.shift()!;
        this.assignJob(worker, message, job);
      } else {
        this.jobQueue.push({ workerMessage: message, job });
      }
    });
  }

  /**
   * Broadcasts a message to all workers (e.g. for initialization).
   */
  public async broadcast(
    type: WorkerMessage["type"],
    payload?: any,
  ): Promise<any[]> {
    this.idleWorkers = [];
    const promises = this.workers.map((worker) => {
      return new Promise((resolve, reject) => {
        const id = this.nextMessageId++;
        const message: WorkerMessage = { id, type, payload };
        const job: Job = { id, resolve, reject };

        // To broadcast correctly, we temporarily skip the normal job queue for this specific message
        // to ensure it goes to a specific worker. By using assignJob, the worker's active job count
        // is incremented, ensuring it isn't marked as idle prematurely.
        this.assignJob(worker, message, job);
      });
    });
    return Promise.all(promises);
  }

  public async terminate() {
    this.jobQueue = [];
    for (const [id, job] of this.activeJobs.entries()) {
      job.reject(new Error("Worker pool terminated."));
    }
    this.activeJobs.clear();
    this.workerActiveJobs.clear();
    await Promise.all(this.workers.map((w) => w.terminate()));
    this.workers = [];
    this.idleWorkers = [];
  }
}
