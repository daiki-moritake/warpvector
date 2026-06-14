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
