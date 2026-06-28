import { wasmBase64 } from "./wasm-binary";

export class WasmWorker {
  public instance: WebAssembly.Instance;
  public memory: WebAssembly.Memory;
  public offset: number;
  public peakOffset: number;
  private readonly defaultOffset = 65536;

  constructor(instance: WebAssembly.Instance) {
    this.instance = instance;
    this.memory = instance.exports.memory as WebAssembly.Memory;
    this.offset = this.defaultOffset;
    this.peakOffset = this.defaultOffset;
  }

  public ensureMemory(requiredBytes: number): boolean {
    if (requiredBytes <= this.memory.buffer.byteLength) {
      return true;
    }

    const currentPages = this.memory.buffer.byteLength / 65536;
    const requiredPages = Math.ceil(requiredBytes / 65536);
    const targetPages = Math.max(requiredPages, currentPages * 2);

    try {
      this.memory.grow(targetPages - currentPages);
      return true;
    } catch (e) {
      if (targetPages > requiredPages) {
        try {
          this.memory.grow(requiredPages - currentPages);
          return true;
        } catch {
          console.warn("WASM memory grow failed", e);
          return false;
        }
      }
      console.warn("WASM memory grow failed", e);
      return false;
    }
  }

  public allocate(bytes: number): number {
    const ptr = this.offset;
    this.offset += bytes;
    if (this.offset % 4 !== 0) {
      this.offset += 4 - (this.offset % 4);
    }
    if (this.offset > this.peakOffset) {
      this.peakOffset = this.offset;
    }
    this.ensureMemory(this.offset);
    return ptr;
  }

  public resetAllocator(): void {
    this.offset = this.defaultOffset;
  }
}

export class WasmPool {
  private module: WebAssembly.Module | null = null;
  private pool: WasmWorker[] = [];
  private currentContext: WasmWorker | null = null;
  private fallbackInstance: WasmWorker | null = null;

  public async init(): Promise<void> {
    if (this.module) return;
    const bytes = Uint8Array.from(atob(wasmBase64), (c) => c.charCodeAt(0));
    this.module = await WebAssembly.compile(bytes);
  }

  public acquire(): WasmWorker {
    if (!this.module) {
      throw new Error("WasmPool not initialized. Call init() first.");
    }
    if (this.pool.length > 0) {
      return this.pool.pop()!;
    }
    // Create new instance
    const instance = new WebAssembly.Instance(this.module);
    return new WasmWorker(instance);
  }

  public release(worker: WasmWorker): void {
    worker.resetAllocator();
    this.pool.push(worker);
  }

  /**
   * 同期処理用：現在のスレッドコンテキスト（擬似）としてのWorkerを設定
   */
  public setCurrentSyncContext(worker: WasmWorker): void {
    this.currentContext = worker;
  }

  public clearCurrentSyncContext(): void {
    this.currentContext = null;
  }

  public getCurrentSyncContext(): WasmWorker | null {
    if (this.currentContext) return this.currentContext;

    // 互換性のため、acquireされずに単発で呼ばれた場合（テストやレガシー呼び出し）は
    // フォールバック用の単一インスタンスを返す
    if (!this.fallbackInstance) {
      if (!this.module) return null;
      this.fallbackInstance = new WasmWorker(
        new WebAssembly.Instance(this.module),
      );
    }
    return this.fallbackInstance;
  }
}

export const globalWasmPool = new WasmPool();
