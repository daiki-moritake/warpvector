export interface IsomorphicWorker {
  postMessage(message: any, transfer?: any[]): void;
  onMessage(callback: (message: any) => void): void;
  onError(callback: (error: Error) => void): void;
  terminate(): Promise<void> | void;
}

/**
 * Creates an isomorphic worker depending on the environment (Node.js or Browser).
 * @param scriptPath The path or URL to the worker script
 */
export function createWorker(scriptPath: string | URL): IsomorphicWorker {
  // Check for Web Worker environment (Browser)
  if (typeof Worker !== "undefined") {
    const worker = new Worker(scriptPath);
    return {
      postMessage: (msg, transfer) => worker.postMessage(msg, transfer || []),
      onMessage: (cb) => {
        worker.onmessage = (e) => cb(e.data);
      },
      onError: (cb) => {
        worker.onerror = (e) => cb(new Error(e.message));
      },
      terminate: () => worker.terminate(),
    };
  }

  // Check for Node.js environment
  if (
    typeof process !== "undefined" &&
    process.versions &&
    process.versions.node
  ) {
    // Dynamically require worker_threads to avoid breaking browser builds

    const { Worker: NodeWorker } = require("worker_threads");
    const worker = new NodeWorker(scriptPath);
    return {
      postMessage: (msg, transfer) => worker.postMessage(msg, transfer),
      onMessage: (cb) => worker.on("message", cb),
      onError: (cb) => worker.on("error", cb),
      terminate: async () => {
        await worker.terminate();
      },
    };
  }

  throw new Error(
    "Unsupported environment: Neither Web Worker nor Node.js worker_threads are available.",
  );
}
