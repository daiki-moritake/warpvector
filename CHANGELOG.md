# Changelog

All notable changes to WarpVector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Test badge updated: 209 → **282 tests** (+35%)
- expect() calls: 1,188 → **1,554** (+31%)

### Testing
- **282 tests** across 42 files (all passing)
- 1,554 expect() calls
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
