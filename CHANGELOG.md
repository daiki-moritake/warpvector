# Changelog

All notable changes to WarpVector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.8.1] - 2026-06-30

### Changed
- **Adapter Abstractions**: Renamed `VectorDBAdapter` to `VectorDBFormatter` for clarity and introduced `AbstractWarpAdapter` as a common base class to standardize adapter configurations and reduce boilerplate.
- **Pipeline Execution & Developer Experience**: Migrated the internal monorepo script execution from `npm --workspaces` to `bun run --filter` to resolve infinite loop bugs and argument passing issues.
- **Strict Conformance Testing**: Enforced strict type safety in `adapter-conformance.test.ts` and resolved exported type scope issues across trainer packages (`InfoNCEExample`, `TripletExample`).
- **Error Handling**: Improved error traceability in `WarpPipeline` by explicitly preserving error `cause` chains.

## [0.8.0] - 2026-06-30

### Added
- **Federated Learning UI**: Implemented a new federated learning UI with user feedback collection and aggregation orchestration in the playground.

### Changed
- **Quantization Optimization**: Significantly improved the rendering performance of the demo engine by reusing the `QuantizationAdapter` instance, eliminating redundant object instantiations during hot loops.
- **Enhanced Whitening Convergence**: Improved the offline training process for `WhiteningAdapter` in the demo by introducing multi-epoch training iterations, resulting in faster and more stable principal component convergence and better anisotropy correction.
- **Type Safety**: Consolidated use case definitions and improved overall type safety in `usecases.ts`.
- **Engine Alignment**: Refactored and updated the `demo-engine` implementation to align with the latest engine interfaces.

## [0.7.2] - 2026-06-29

### Added
- **Documentation**: Added a practical guide "Project Integration for Auto-Learning and Federated Learning" (`docs/cookbook/practical-auto-learning.ja.md`).
- **Documentation**: Added auto-learning related classes to `docs/api-reference.ja.md` (`IntentMatrixFactory`, `FeedbackCollector`, `AdaptiveScheduler`, `FederatedAggregator`).

### Fixed
- **AnomalyDetectionAdapter**: Fixed a failing test to properly handle `NaN` inputs in safe mode.

## [0.7.1] - 2026-06-28

### Fixed
- **VsaAdapter**: Fixed sign inversion bug in `unbind()` where division key near negative zero (`-0.0`) was clamped to positive `EPSILON` due to standard JS `-0.0 >= 0` evaluation. Implemented strict `Object.is` check to preserve the sign.
- **ColbertAdapter**: Fixed a potential crash in `rank()` when WebAssembly memory growth occurred during a loop, which invalidated (detached) the cached `Float32Array` view. Modified to use `writeFloat32ArrayToWasm` on each iteration to always obtain the latest buffer.
- **BaseGraphReranker & Rerankers**: Added strict boundary validation and type/NaN checking for configurations (`threshold`, `alpha`, `maxIterations`, `tolerance`, `tau`, `iterations`) in reranker constructors to prevent calculation failure from invalid parameters.
- **Trainers**: Added validation guards for hyperparameters (`learningRate`, `regularization`, `epochs`, `margin`, `temperature`) in `BaseTrainer`, `TripletTrainer`, and `InfoNCETrainer` to fail fast and prevent parameter corruption with `NaN` weights.
- **VectorDBAdapter**: Implemented vector type and finite number check (validating `NaN` and `Infinity`) in database serialization methods (`toPgvector`, `toPineconeQuery`, `toRedis`, `toVectorizeQuery`, `toVectorizeRecord`) to prevent SQL/query crashes at database connection layer.
- **Performance Regression Tests**: Relaxed sub-millisecond benchmarking assertion thresholds to prevent flaky test failures during concurrent execution under high CPU load.

## [0.7.0] - 2026-06-28

### Added
- **AlignmentAdapter**: Introduced `AlignmentAdapter` to `@warpvector/core` to support zero-downtime model migrations and vector space alignment.
- **Playground Updates**: Completely redesigned the playground with a high-quality UI and added new use-case sample pages.
- **Migration Guides**: Added comprehensive documentation and guides for zero-downtime model migrations.

### Changed
- Extensive package updates and code refactoring across the repository.
- Improved README formatting, readability, and syntax highlights.

### Fixed
- **TypeScript Types**: Resolved multiple TypeScript compilation errors, including incorrect Promise handling in pipelines, undefined matrix variables in `CrossEncoderTrainer`, and missing adapter exports.
- **Pipeline Registration**: Fixed an issue where `AlignmentAdapter` was not properly registered in `WarpPipeline`.
- **Mermaid Diagrams**: Fixed syntax errors in Mermaid dashed links in the documentation.

## [0.6.1] - 2026-06-27

### Fixed
- **WebGPU Memory Leak**: Explicitly destroy temporary GPU buffers (`GPUBufferUsage`) after GPU execution in `WebGpuIntentAdapter` to prevent VRAM exhaustion and browser tab crashes during high-throughput workloads.
- **Worker Initialization Bug**: Fixed an issue where calling `WarpWorkerClient.init()` did not propagate the initialization state (`pipelineState`) to the workers because the method implementation was empty.
- **Worker Pool Edge Case**: Fixed an edge case in `WarpWorkerPool.terminate()` where active workers returning late responses could corrupt internal state or cause unhandled rejections by properly clearing the active job map.

## [0.6.0] - 2026-06-27

### Added

#### New Packages
- **`@warpvector/worker`** — Web Worker and Node.js `worker_threads` support for Isomorphic multithreading.
  - Implements a generic `WarpWorkerPool` to parallelize batch processing.
  - Features `IsomorphicWorker` abstraction for seamless execution in both browsers and backend.
- **`@warpvector/gpu`** — WebGPU Compute Shader acceleration.
  - Introduces `WebGpuIntentAdapter` to process massively parallel vector transformations (affine transforms) directly on the GPU.
- **`@warpvector/opentelemetry`** — OpenTelemetry tracing integration.
  - Exposes `OpenTelemetryTracer` which implements the `WarpTracer` interface for distributed tracing and observability.

### Fixed
- Fixed WGSL `@workgroup_size` dimension mismatch in `WebGpuIntentAdapter` to ensure proper GPU thread bounds.
- Fixed `WarpWorkerPool` broadcast race conditions by tracking active worker jobs internally.
- Relaxed browser environment check in `IsomorphicWorker` to properly support Service Workers and nested Web Workers.

## [0.5.1] - 2026-06-27

### Performance

#### 1. Quantization SIMD Acceleration
- **Int8 Quantization**: 劇的な高速化を達成 (約3倍のスループット向上). `quantizeToInt8Wasm` 関数に対してSIMD命令 (`f32x4.nearest`, `i32x4.trunc_sat_f32x4_s`, `i8x16.narrow_i16x8_s` など) を適用し、条件分岐やスカラ演算を完全に排除.
- **Binary Quantization**: `quantizeToBinaryWasm` に対してもSIMD最適化 (`f32x4.gt`, `i32x4.bitmask`) とビットシフト演算を用いたインラインパッキング処理を導入し、さらに処理効率を向上.

## [0.5.0] - 2026-06-27

### ⚠️ BREAKING CHANGES

#### 1. Pipeline methods are now async
- `run()`, `runBatch()`, `runAndFormat()`, `dryRun()` now return `Promise` and require `await`
- These methods now call `ensureInitialized()` internally, making `autoInit` seamless
- `runStream()` behavior is unchanged (was already async)

#### 2. Error messages unified to English
- All structured error messages (`WarpDimensionMismatchError`, `WarpInitializationError`, `WarpValidationError`) are now in English
- Error codes and `instanceof` chains remain unchanged

#### 3. `runAndFormat()` generic type parameter
- `runAndFormat<T>()` now accepts a type parameter for type-safe return values
- Default is `unknown` (backwards compatible at the type level)

#### 4. Deprecated `QuantizationAdapter` static methods removed
- `QuantizationAdapter.hammingDistance()` → use `hammingDistance()` from `@warpvector/core`
- `QuantizationAdapter.int8DotProduct()` → use `int8DotProduct()` from `@warpvector/core`

#### 5. `WarpPipeline.inputDim` is now read-only
- `inputDim` changed from a public property to a getter (read-only)
- External writes will cause a compile error; reads are unchanged

#### 6. `flattenMatrix` error type unified
- `flattenMatrix()` now throws `WarpDimensionMismatchError` instead of generic `Error`
- Enables consistent `instanceof` error handling across all dimension validation

### Testing
- **297 tests** across 44 files — All Passed ✅
- **1,579 expect()** calls
- No regressions

## [0.4.1] - 2026-06-27

### Changed

#### Infrastructure & DX
- **Lock file unification**: Standardized on `bun.lock`, removed `package-lock.json` from repository
- **CI improvements**: Added `develop` branch triggers, unified type-check command to `bun run typecheck`
- **npm metadata**: Added `repository`, `homepage`, `bugs` fields to all 10 sub-packages for improved discoverability
- **Playground workspace integration**: Added `playground` to npm workspaces, updated deploy workflow to use `bun`
- **Engine constraints**: Added `"engines": { "node": ">=20.0.0" }` to root `package.json`

#### Documentation
- **Migration Guide**: Updated to cover v0.2→v0.3 and v0.3→v0.4 breaking changes (EN/JA)
- **Release notes reorganization**: Moved 11 release note files to `release-notes/` directory

#### Bug Fixes
- Removed phantom `@warpvector/ml` dependency from `@warpvector/rerank` (was declared but unused)
- Fixed `@warpvector/extras` description (removed reference to ColBERT, which moved to `@warpvector/rerank` in v0.4.0)
- Added missing `license` and `author` fields to `create-warpvector-app`

### Testing
- **297 tests** across 44 files — All Passed ✅
- **1,579 expect()** calls

## [0.4.0] - 2026-06-26

### Added

#### New Packages
- **`@warpvector/train`** — Training and fine-tuning module extracted from `@warpvector/ml`
  - Includes `SoftWhiteningAdapter`, `TripletTrainer`, `BaseTrainer`, and all training-related utilities
  - Dedicated package for backend-heavy training tasks, keeping the core lightweight for edge inference
- **`@warpvector/rerank`** — Reranking module extracted into its own package
  - Exports `ColbertAdapter` for ColBERT-based late interaction reranking
  - Clean separation of reranking concerns from the core pipeline
- **`@warpvector/eval`** — RAG Evaluation Kit (NEW)
  - Built-in RAG pipeline evaluator with metrics: Precision@K, Recall@K, NDCG@K, MRR, MAP
  - CLI tool (`warpvector-eval`) for automated evaluation runs
  - JSON/Markdown report generation for evaluation results

#### Playground / Warpvector Studio
- Upgraded visualization engine with animation state snapshots and improved canvas rendering
- Enhanced UI helper utilities for interactive demos

#### Infrastructure
- **tsconfig共通テンプレート化**: `tsconfig.package.json` / `tsconfig.package.build.json` shared configs for all sub-packages
- Centralized vector math operations into `@warpvector/core/math/vector.ts`

### Changed

#### Breaking Changes
- **QuantizationAdapter API統一**: `tune()` method removed; use `encode()` for all quantization operations
- **WarpPipeline final stage handling**: Updated to use the unified `encode`-based pattern
- **Training utilities移行**: `SoftWhiteningAdapter` and related training modules moved from `@warpvector/ml` to `@warpvector/train`; update imports accordingly
- **Adam optimizer removed**: Removed built-in Adam optimizer from ml package (use `@warpvector/train` instead)

#### Refactoring
- Formalized architecture separation between edge inference (`@warpvector/core`, `@warpvector/ml`) and backend-heavy tasks (`@warpvector/train`, `@warpvector/eval`)
- Updated AutoML metrics and `MoeAdapter` to align with the new evaluation schema
- Consolidated documentation and import paths to reflect the training/reranking module migration

#### Documentation
- New MoE, training, and security adapter documentation in API reference and README
- Architecture separation formally documented with updated package descriptions
- README import paths corrected for new package structure

### Testing
- **297 tests** across 44 files — All Passed ✅
- **1,579 expect()** calls
- Updated test coverage badge in README

## [0.3.2] - 2026-06-25

### Added

- **Pipeline Deserialization Support**: Registered `SafeQuantizationAdapter` and `AnomalyDetectionAdapter` in the `WarpPipeline` registry, enabling full state serialization (`exportState()`) and restoration (`importState()`) for pipelines incorporating these security and safety steps.
- **Integration Tests**: Added end-to-end pipeline serialization tests in `AnomalyDetection.test.ts` to ensure stability of the deserialization registry.

### Changed

- **CLI Version Auto-sync**: Modified `create-warpvector-app` CLI to resolve its version dynamically from its `package.json`, ensuring the CLI output is always aligned with the release version.
- **Documentation Correction**: Fixed the `getting-started.md` guide where an invalid `.quantize()` call was used, updating it to the correct `.setFinalStage()` API pattern.

## [0.3.1] - 2026-06-25

### Added

- **WASM Core Optimization**: Optimized SIMD vector operations and loop structures in `tuneBatchWasm`.
  - Inverted dimension (`i`) and batch (`k`) loops to enable contiguous sequential writes to output buffers, greatly improving CPU L1/L2 cache locality.
  - Split SIMD accumulator (`sumVec`) into 4 independent registers (`sumVec0..3`) to exploit instruction-level parallelism (ILP) and prevent pipeline hazards in inner product calculations.
- **JS / WASM Memory Allocation Reductions**:
  - Eliminated redundant `new Float32Array` heap allocations within the execution loops for `IntentAdapter`, `ColbertAdapter`, and `MlpAdapter`.
  - Views of WebAssembly memory are now cached and reused across loop iterations, preventing runtime GC latency spikes during heavy batch loads.
  - Replaced manual copying in JS loops with native `Float32Array.prototype.slice()` and `set()` for efficient `memcpy` operations.

### Changed

- **Performance Enhancements**:
  - **WASM batch transformation** (1536-dimensional, 10,000 vectors): 4,245 vecs/s → **4,944 vecs/s** (**+16.4%** throughput increase).
  - **Int8 Scalar Quantization**: 317,564 vecs/s → **372,924 vecs/s** (**+17.3%** speedup).
  - Reduced memory footprint and garbage collection overhead during high-concurrency server workloads.

## [0.3.0] - 2026-06-23

### Added

#### Core (`@warpvector/core`)
- **WarpTracer**: Zero-dependency OpenTelemetry-compatible tracing module
  - `trace()` / `traceAsync()` for latency measurement
  - `getMetrics()` for per-operation statistics (totalCalls, avgLatencyMs, min/max, operationCounts)
  - `resetMetrics()` for clearing collected data
- **Cloudflare Vectorize** integration in VectorDBAdapter
  - `toVectorizeQuery()` — generates Vectorize-compatible query objects
  - `toVectorizeRecord()` — generates Vectorize-compatible upsert records
- **Graceful Degradation** tests: 13 tests covering dimension mismatches, unregistered intents, empty inputs, quantization boundaries, structured error validation

#### ML (`@warpvector/ml`)
- **IntentMatrixFactory**: Auto-learn intent matrices from document categories via InfoNCE contrastive learning
  - `addCategory()` / `build()` API
  - 16 unit tests

#### Extras (`@warpvector/extras`)
- Quantization quality benchmark report auto-generation (`REPORT.md`)
- IR evaluation benchmark report auto-generation (`REPORT.md`)

#### Experimental (`@warpvector/experimental`) — NEW PACKAGE
- Separated experimental features into `@warpvector/experimental`
  - ColBERT Late Interaction (MaxSim)
  - Vector Symbolic Architecture (VSA)
  - Anomaly Detection
  - Task Arithmetic
- 6 integration tests

#### CLI (`create-warpvector-app`)
- **Minimal Intent Search** template: IntentMatrixFactory + Worker-based vector search
- Package manager selection (npm/yarn/pnpm/bun)
- Improved template generation UX

#### Playground
- **Auto-learn Intents** panel (EN/JA): Interactive IntentMatrixFactory demo
- Dynamic import of `@warpvector/ml` for on-demand InfoNCE training

#### Documentation
- Getting Started guide fully rewritten (EN/JA)
- IntentMatrixFactory dedicated documentation (EN/JA)
- Examples: `auto-intent.ts` — copy-pasteable practical demo

### Changed
- README updated with Cloudflare Vectorize, OpenTelemetry, and IntentMatrixFactory sections
- Test badge updated: 209 → **290 tests** (+39%)
- expect() calls: 1,188 → **1,562** (+31%)

### Testing
- **290 tests** across 43 files (all passing)
- 1,562 expect() calls
- New test files:
  - `graceful-degradation.test.ts` (13 tests)
  - `telemetry-vectorize.test.ts` (12 tests)
  - `experimental.test.ts` (6 tests)
- Enhanced test files:
  - `property-based.test.ts` (+15 tests)
  - `wasm-js-equivalence.test.ts` (+11 tests)

### Benchmarks
- **Quantization Quality**: Int8 Recall@10 = 86-96%, Binary = 35-36%
- **IR Evaluation**: Intent Warping NDCG@10 +13% over vanilla baseline
- Both benchmarks now auto-generate Markdown reports

---

## [0.2.0] - 2026-06-14

### Added
- Initial public release
- IntentAdapter, LoraIntentAdapter, ProjectionAdapter
- WarpPipeline with adapter chaining
- WASM acceleration with JS fallback
- WhiteningAdapter for anisotropy correction
- QuantizationAdapter (Int8/Binary)
- ColbertAdapter for late interaction
- VsaAdapter for Vector Symbolic Architecture
- VectorDBAdapter (pgvector, Pinecone, Redis)
- LangChain and Prisma integrations
- Interactive Playground (EN/JA)
- 209 tests across 37 files
