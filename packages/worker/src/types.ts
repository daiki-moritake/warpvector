export type WorkerMessageType = "init" | "runBatch" | "runStream";

export interface WorkerMessage {
  id: number;
  type: WorkerMessageType;
  payload?: any;
}

export interface WorkerResponse {
  id: number;
  success: boolean;
  payload?: any;
  error?: string;
}

export interface Job {
  id: number;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
}
