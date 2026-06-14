import { wasmBase64 } from "./wasm-binary";

let wasmInstance: WebAssembly.Instance | null = null;
let wasmMemory: WebAssembly.Memory | null = null;
let initPromise: Promise<WebAssembly.Instance> | null = null;

export function getWasmMemory(): WebAssembly.Memory | null {
  return wasmMemory;
}

export function getWasmInstance(): WebAssembly.Instance | null {
  return wasmInstance;
}

export async function initWasm(): Promise<WebAssembly.Instance | null> {
  if (wasmInstance) return wasmInstance;
  if (initPromise) return initPromise;

  try {
    const bytes = Uint8Array.from(atob(wasmBase64), (c) => c.charCodeAt(0));
    const module = await WebAssembly.compile(bytes);

    wasmInstance = new WebAssembly.Instance(module);

    wasmMemory = wasmInstance.exports.memory as WebAssembly.Memory;
    return wasmInstance;
  } catch (e) {
    console.warn("WASM initialization failed, falling back to JS.", e);
    return null;
  }
}

export function ensureWasmMemory(requiredBytes: number): boolean {
  if (!wasmMemory) return false;

  if (requiredBytes <= wasmMemory.buffer.byteLength) {
    return true;
  }

  const currentPages = wasmMemory.buffer.byteLength / 65536;
  const requiredPages = Math.ceil(requiredBytes / 65536);
  try {
    wasmMemory.grow(requiredPages - currentPages);
    return true;
  } catch (e) {
    console.warn("WASM memory grow failed", e);
    return false;
  }
}

let globalOffset = 65536; // 最初の64KBはシステム予約領域として保護する

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
