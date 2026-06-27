import { wasmBase64 } from "./wasm-binary";

let wasmInstance: WebAssembly.Instance | null = null;
let wasmMemory: WebAssembly.Memory | null = null;
let initPromise: Promise<WebAssembly.Instance | null> | null = null;

export function getWasmMemory(): WebAssembly.Memory | null {
  return wasmMemory;
}

export function getWasmInstance(): WebAssembly.Instance | null {
  return wasmInstance;
}

export async function initWasm(): Promise<WebAssembly.Instance | null> {
  if (wasmInstance) return wasmInstance;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      const bytes = Uint8Array.from(atob(wasmBase64), (c) => c.charCodeAt(0));
      const module = await WebAssembly.compile(bytes);

      wasmInstance = new WebAssembly.Instance(module);

      wasmMemory = wasmInstance.exports.memory as WebAssembly.Memory;
      return wasmInstance;
    } catch (e) {
      console.warn("WASM initialization failed, falling back to JS.", e);
      initPromise = null; // 失敗時はリトライ可能にする
      return null;
    }
  })();

  return initPromise;
}

export function ensureWasmMemory(requiredBytes: number): boolean {
  if (!wasmMemory) return false;

  if (requiredBytes <= wasmMemory.buffer.byteLength) {
    return true;
  }

  const currentPages = wasmMemory.buffer.byteLength / 65536;
  const requiredPages = Math.ceil(requiredBytes / 65536);
  // 指数的成長戦略: 必要量の2倍、または現在の2倍のうち大きい方を確保
  // これにより頻繁な grow() 呼び出しを回避し、大規模バッチ処理のパフォーマンスを改善
  const targetPages = Math.max(requiredPages, currentPages * 2);
  try {
    wasmMemory.grow(targetPages - currentPages);
    return true;
  } catch (e) {
    // 2倍が無理な場合は必要最小限で再試行
    if (targetPages > requiredPages) {
      try {
        wasmMemory.grow(requiredPages - currentPages);
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

let globalOffset = 65536; // 最初の64KBはシステム予約領域として保護する
let peakOffset = 65536;

/**
 * WASMメモリの使用統計情報
 */
export interface WasmMemoryStats {
  /** 現在使用中のバイト数（アロケータのオフセット） */
  usedBytes: number;
  /** WASMメモリの総バイト数（確保済みページ分） */
  totalBytes: number;
  /** 過去の最大使用バイト数 */
  peakBytes: number;
}

/**
 * WASMメモリの使用統計を取得します。
 * 運用中のメモリ監視やデバッグに使用してください。
 *
 * @example
 * ```typescript
 * const stats = getWasmMemoryStats();
 * console.log(`Used: ${stats.usedBytes}, Peak: ${stats.peakBytes}, Total: ${stats.totalBytes}`);
 * ```
 */
export function getWasmMemoryStats(): WasmMemoryStats {
  return {
    usedBytes: globalOffset,
    totalBytes: wasmMemory ? wasmMemory.buffer.byteLength : 0,
    peakBytes: peakOffset,
  };
}

/**
 * 競合しないWASMメモリ領域を確保する簡易アロケータ
 */
export function allocateWasmMemory(bytes: number): number {
  const ptr = globalOffset;
  globalOffset += bytes;
  // 4バイトアラインメント
  if (globalOffset % 4 !== 0) {
    globalOffset += 4 - (globalOffset % 4);
  }
  // ピーク使用量を更新
  if (globalOffset > peakOffset) {
    peakOffset = globalOffset;
  }
  ensureWasmMemory(globalOffset);
  return ptr;
}

/**
 * アロケータのオフセットをリセットする（テスト等で使用）
 */
export function resetWasmAllocator(): void {
  globalOffset = 65536;
}

export function getWasmAllocatorOffset(): number {
  return globalOffset;
}

export function setWasmAllocatorOffset(offset: number): void {
  globalOffset = offset;
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
    // memmove 相当の高速コピー
    f32.set(data, floatOffset);
  } else {
    for (let i = 0; i < data.length; i++) {
      f32[floatOffset + i] = data[i];
    }
  }
}

/**
 * 非同期処理でのWASMメモリ競合を防ぐための排他制御用ミューテックスクラス
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
  const initialOffset = getWasmAllocatorOffset();
  try {
    return fn();
  } finally {
    setWasmAllocatorOffset(initialOffset);
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
