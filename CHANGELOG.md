# Changelog

All notable changes to WarpVector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
