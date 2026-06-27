import { wasmBase64 } from "./wasm-binary";

import { globalWasmPool } from "./WasmPool";

let initPromise: Promise<WebAssembly.Instance | null> | null = null;

export function getWasmMemory(): WebAssembly.Memory | null {
  const ctx = globalWasmPool.getCurrentSyncContext();
  return ctx ? ctx.memory : null;
}

export function getWasmInstance(): WebAssembly.Instance | null {
  const ctx = globalWasmPool.getCurrentSyncContext();
  return ctx ? ctx.instance : null;
}

export async function initWasm(): Promise<WebAssembly.Instance | null> {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      await globalWasmPool.init();
      // Ensure fallback instance is created if needed
      return getWasmInstance();
    } catch (e) {
      console.warn("WASM initialization failed, falling back to JS.", e);
      initPromise = null;
      return null;
    }
  })();

  return initPromise;
}

export function ensureWasmMemory(requiredBytes: number): boolean {
  const ctx = globalWasmPool.getCurrentSyncContext();
  if (!ctx) return false;
  return ctx.ensureMemory(requiredBytes);
}

/**
 * WASMメモリの使用統計情報
 */
export interface WasmMemoryStats {
  usedBytes: number;
  totalBytes: number;
  peakBytes: number;
}

export function getWasmMemoryStats(): WasmMemoryStats {
  const ctx = globalWasmPool.getCurrentSyncContext();
  if (!ctx) {
    return { usedBytes: 0, totalBytes: 0, peakBytes: 0 };
  }
  return {
    usedBytes: ctx.offset,
    totalBytes: ctx.memory.buffer.byteLength,
    peakBytes: ctx.peakOffset,
  };
}

/**
 * 競合しないWASMメモリ領域を確保する簡易アロケータ
 */
export function allocateWasmMemory(bytes: number): number {
  const ctx = globalWasmPool.getCurrentSyncContext();
  if (!ctx) {
    throw new Error("WASM context not found. Call initWasm() first.");
  }
  return ctx.allocate(bytes);
}

export function resetWasmAllocator(): void {
  const ctx = globalWasmPool.getCurrentSyncContext();
  if (ctx) ctx.resetAllocator();
}

export function getWasmAllocatorOffset(): number {
  const ctx = globalWasmPool.getCurrentSyncContext();
  return ctx ? ctx.offset : 0;
}

export function setWasmAllocatorOffset(offset: number): void {
  const ctx = globalWasmPool.getCurrentSyncContext();
  if (ctx) ctx.offset = offset;
}

/**
 * データをWASMメモリに書き込むヘルパー関数
 */
export function writeFloat32ArrayToWasm(
  memory: WebAssembly.Memory,
  data: number[] | Float32Array,
  byteOffset: number,
): void {
  const f32 = new Float32Array(memory.buffer);
  const floatOffset = byteOffset / 4;

  if (data instanceof Float32Array) {
    f32.set(data, floatOffset);
  } else {
    for (let i = 0; i < data.length; i++) {
      f32[floatOffset + i] = data[i];
    }
  }
}

/**
 * 非同期処理でのWASMメモリ競合を防ぐための排他制御用ミューテックスクラス
 * @deprecated WasmPool により不要になりました。後方互換性のために残しています。
 */
export class WasmMutex {
  private queue: Promise<void> = Promise.resolve();

  public async runExclusive<T>(fn: () => Promise<T> | T): Promise<T> {
    const next = this.queue.then(async () => {
      return fn();
    });
    this.queue = next.then(
      () => {},
      () => {},
    );
    return next;
  }
}

export const wasmMutex = new WasmMutex();

/**
 * WASMメモリスタックのオフセットを自動的に管理（確保と解放）するヘルパー関数
 */
export function withWasmMemoryStack<T>(fn: () => T): T {
  const ctx = globalWasmPool.getCurrentSyncContext();
  if (!ctx) return fn();
  
  const initialOffset = ctx.offset;
  try {
    return fn();
  } finally {
    ctx.offset = initialOffset;
  }
}

/**
 * WASMメモリからFloat32Arrayを読み取るヘルパー関数
 */
export function readFloat32ArrayFromWasm(
  memory: WebAssembly.Memory,
  byteOffset: number,
  length: number,
): Float32Array {
  const f32 = new Float32Array(memory.buffer);
  const floatOffset = byteOffset / 4;
  return new Float32Array(f32.subarray(floatOffset, floatOffset + length));
}
