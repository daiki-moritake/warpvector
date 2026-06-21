# Contributing to WarpVector

Thank you for your interest in contributing to WarpVector! This guide will help you get started.

## 🚀 Quick Setup

```bash
# Clone the repository
git clone https://github.com/daiki-moritake/warpvector.git
cd warpvector

# Install dependencies (Bun recommended)
bun install

# Build WASM + all packages
bun run build

# Run tests
bun test

# Run benchmarks
bun run bench
```

## 📁 Project Structure

```
warpvector/
├── packages/
│   ├── core/       # Zero-dependency core (IntentAdapter, Pipeline, WASM)
│   ├── ml/         # ML adapters (MLP, Whitening, Trainers)
│   ├── extras/     # Quantization, ColBERT, Fusion, VSA
│   ├── prisma/     # Prisma + pgvector integration
│   └── langchain/  # LangChain integration
├── examples/       # Runnable example scripts
├── benchmarks/     # Performance benchmarks
├── docs/           # Documentation
└── assembly/       # AssemblyScript WASM source
```

## 🔧 Development Workflow

### Making Changes

1. **Fork** the repository and create a feature branch
2. Make your changes in the appropriate package
3. Add tests for new functionality
4. Ensure all tests pass: `bun test`
5. Ensure the build succeeds: `bun run build`
6. Submit a Pull Request

### Running Specific Tests

```bash
# All tests
bun test

# Specific package
bun test packages/core/tests/

# Specific file
bun test packages/core/tests/validation.test.ts
```

### Code Style

- TypeScript strict mode
- No external dependencies in `@warpvector/core`
- All public APIs must have JSDoc with `@example` blocks
- Error messages should use structured `WarpError` subclasses

## 📝 Types of Contributions

### 🐛 Bug Reports
- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include reproduction steps and environment details

### 💡 Feature Requests
- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the use case and expected behavior

### 🧪 Code Contributions

**Good first issues:**
- Adding `@example` blocks to JSDoc comments
- Writing tests for edge cases
- Improving error messages
- Adding new integration adapters (Supabase, Drizzle, etc.)

**Intermediate:**
- New adapter implementations
- Performance optimizations
- Documentation improvements

**Advanced:**
- WASM kernel optimizations (AssemblyScript)
- New training algorithms
- New reranking strategies

## 🏗 Adding a New Adapter

1. Create your adapter in the appropriate package (core/ml/extras)
2. Implement the `WarpAdapter` or `FinalStageAdapter` interface
3. Add `exportState()` for serialization support
4. Register it in `AdapterRegistry` for `importState` support
5. Add comprehensive tests
6. Add documentation in `docs/`
7. Add an example in `examples/`

```typescript
import { WarpAdapter, InputVector, OutputVector } from "@warpvector/core";

export class MyAdapter implements WarpAdapter {
  tune(vector: InputVector, context?: string): OutputVector {
    // Your transformation logic
  }
  
  exportState(): string {
    // Serialize state to JSON string
  }
  
  static importState(state: string): MyAdapter {
    // Restore from serialized state
  }
}
```

## 📋 Pull Request Checklist

- [ ] Tests added/updated and passing (`bun test`)
- [ ] Build succeeds (`bun run build`)
- [ ] JSDoc added for public APIs
- [ ] Documentation updated (if applicable)
- [ ] No new external dependencies in `@warpvector/core`
- [ ] Commit messages are clear and descriptive

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
