import { WarpWorkerHandler } from "../src/WarpWorkerHandler";

// Mock adapter registration to simulate global registry
import { WarpPipeline, InputVector, TransformOutput, WarpAdapter } from "@warpvector/core";

// We need a dummy adapter to test with
class DummyAdapter implements WarpAdapter {
  constructor(private factor: number) {}
  tune(vector: InputVector, intent?: string): TransformOutput {
    const f32 = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
      f32[i] = (vector[i] as number) * this.factor;
    }
    return f32;
  }
  exportState() {
    return { factor: this.factor };
  }
  static importState(state: any) {
    return new DummyAdapter(state.factor);
  }
}

WarpPipeline.registerAdapter("DummyAdapter", (state) => DummyAdapter.importState(state));

const handler = new WarpWorkerHandler();
handler.listen();
