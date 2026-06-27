/// <reference types="@webgpu/types" />
import { WarpAdapter, InputVector, TransformOutput, AdapterState } from "@warpvector/core";

export interface IntentWeights {
  matrix: number[][] | number[];
  bias: number[];
}

export class WebGpuIntentAdapter implements WarpAdapter {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;

  constructor(
    private intents: Record<string, IntentWeights>,
    private inputDim: number,
    private outputDim: number
  ) {}

  public async init(): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      console.warn("WebGPU is not supported on this device/browser.");
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.warn("Failed to request WebGPU adapter.");
      return;
    }

    this.device = await adapter.requestDevice();

    const shaderCode = `
      struct Matrix {
        size: vec2<f32>,
        numbers: array<f32>,
      }

      struct Vector {
        size: f32,
        numbers: array<f32>,
      }

      @group(0) @binding(0) var<storage, read> inputVectors: Vector; // batched input
      @group(0) @binding(1) var<storage, read> weightMatrix: Matrix; // weights for specific intent
      @group(0) @binding(2) var<storage, read> biasVector: Vector; // bias for specific intent
      @group(0) @binding(3) var<storage, read_write> outputVectors: Vector; // batched output

      // Parameters: x = vector index, y = output dimension index
      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let vectorIdx = global_id.x;
        let dimIdx = global_id.y;

        let inputDim = u32(weightMatrix.size.x);
        let outputDim = u32(weightMatrix.size.y);
        let batchSize = u32(inputVectors.size) / inputDim;

        if (vectorIdx >= batchSize || dimIdx >= outputDim) {
          return;
        }

        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < inputDim; i = i + 1u) {
          sum = sum + inputVectors.numbers[vectorIdx * inputDim + i] * weightMatrix.numbers[dimIdx * inputDim + i];
        }
        
        sum = sum + biasVector.numbers[dimIdx];
        outputVectors.numbers[vectorIdx * outputDim + dimIdx] = sum;
      }
    `;

    const shaderModule = this.device.createShaderModule({ code: shaderCode });
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  tune(vector: InputVector, context?: string): TransformOutput {
    // Synchronous execution is not possible with WebGPU, we must fallback to a JS loop or throw.
    // For single vector, we could fallback, but since WebGPU is async, WarpPipeline will use tuneBatchAsync.
    throw new Error("WebGpuIntentAdapter requires tuneBatchAsync. Call runBatch() on WarpPipeline.");
  }

  public async tuneBatchAsync(vectors: InputVector[], context?: string): Promise<TransformOutput[]> {
    if (!this.device || !this.pipeline) {
      throw new Error("WebGPU is not initialized or not supported. Make sure to call init() and that your environment supports WebGPU.");
    }

    const intent = context || "default";
    const weights = this.intents[intent];
    if (!weights) {
      // Return original vectors if intent not found
      return vectors.map(v => v instanceof Float32Array ? v : new Float32Array(v));
    }

    const batchSize = vectors.length;
    
    // Prepare flattened matrix
    const flatMatrix = new Float32Array(this.inputDim * this.outputDim);
    if (Array.isArray(weights.matrix[0])) {
      for (let i = 0; i < this.outputDim; i++) {
        for (let j = 0; j < this.inputDim; j++) {
          flatMatrix[i * this.inputDim + j] = (weights.matrix as number[][])[i][j];
        }
      }
    } else {
      flatMatrix.set(weights.matrix as number[]);
    }
    
    const bias = new Float32Array(weights.bias);
    
    // Prepare input buffer
    const flatInput = new Float32Array(batchSize * this.inputDim);
    for (let i = 0; i < batchSize; i++) {
      flatInput.set(vectors[i], i * this.inputDim);
    }

    // Input Buffer
    const inputBuffer = this.device.createBuffer({
      size: flatInput.byteLength + 4, // +4 for size field
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(inputBuffer, 0, new Float32Array([flatInput.length]));
    this.device.queue.writeBuffer(inputBuffer, 4, flatInput);

    // Matrix Buffer
    const matrixBuffer = this.device.createBuffer({
      size: flatMatrix.byteLength + 8, // +8 for size.x, size.y
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(matrixBuffer, 0, new Float32Array([this.inputDim, this.outputDim]));
    this.device.queue.writeBuffer(matrixBuffer, 8, flatMatrix);

    // Bias Buffer
    const biasBuffer = this.device.createBuffer({
      size: bias.byteLength + 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(biasBuffer, 0, new Float32Array([bias.length]));
    this.device.queue.writeBuffer(biasBuffer, 4, bias);

    // Output Buffer
    const outputBufferSize = batchSize * this.outputDim * 4 + 4;
    const outputBuffer = this.device.createBuffer({
      size: outputBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: matrixBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupCountX = Math.ceil(batchSize / 8);
    const workgroupCountY = Math.ceil(this.outputDim / 8);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    passEncoder.end();

    // Map output buffer to read results
    const readBuffer = this.device.createBuffer({
      size: outputBufferSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBufferSize);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = readBuffer.getMappedRange();
    const outputData = new Float32Array(arrayBuffer.slice(4)); // skip size field
    readBuffer.unmap();

    // Destroy all temporary buffers to prevent GPU memory leaks
    inputBuffer.destroy();
    matrixBuffer.destroy();
    biasBuffer.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();

    const results: TransformOutput[] = [];
    for (let i = 0; i < batchSize; i++) {
      results.push(new Float32Array(outputData.buffer, i * this.outputDim * 4, this.outputDim));
    }

    return results;
  }

  exportState(): AdapterState {
    return {
      intents: this.intents,
      inputDim: this.inputDim,
      outputDim: this.outputDim,
    };
  }

  static importState(state: any): WebGpuIntentAdapter {
    return new WebGpuIntentAdapter(state.intents, state.inputDim, state.outputDim);
  }
}
