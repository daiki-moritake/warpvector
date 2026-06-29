# WarpVector Agent Rules

## AssemblyScript vs TypeScript
- The `assembly/` directory contains AssemblyScript code, not standard TypeScript.
- AssemblyScript uses decorators (like `@inline`) and types (like `f32`, `usize`) that are not valid in standard TypeScript.
- **Rule**: Always exclude the `assembly/` directory from standard TypeScript and JavaScript tooling. This includes ESLint (`eslint.config.mjs`), TypeScript compiler (`tsconfig.json`), and Prettier. Do not attempt to fix "Parsing errors" or "Type errors" in `assembly/` using standard TS/JS rules.
