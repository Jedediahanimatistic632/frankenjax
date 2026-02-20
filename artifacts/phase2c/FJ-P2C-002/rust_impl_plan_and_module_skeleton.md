# FJ-P2C-002 Rust Implementation Plan + Module Boundary Skeleton (bd-3dl.13.4)

Date: 2026-02-20
Packet: `FJ-P2C-002` (API transform front-door: jit/grad/vmap)

## 1. Architecture Decision

Decision: Add `fj-api` as a thin user-facing facade crate that wraps the internal dispatch machinery.

- `fj-api` owns the public transform API: `jit()`, `grad()`, `vmap()`, `value_and_grad()`.
- `fj-api` constructs `TraceTransformLedger` and `DispatchRequest` internally; users never see these types.
- `fj-dispatch` remains the execution engine; `fj-api` is a contract-compliant wrapper.
- A top-level `frankenjax` re-export crate is deferred until the API surface stabilizes across multiple packets.

Why this is least-risk now:
- `fj-dispatch` is already tested and stable for all supported transform stacks.
- Adding a thin facade avoids changing any internal crate contracts.
- User-facing error types can wrap `DispatchError` without leaking internal boundaries.
- The `fj-api` → `fj-dispatch` → `fj-core` dependency chain is clean and unidirectional.

## 2. Module Boundary Skeleton

### 2.1 `fj-api` (new crate)

```
crates/fj-api/
  Cargo.toml
  src/
    lib.rs          → re-exports: jit, grad, vmap, value_and_grad, ApiError
    transforms.rs   → JitWrapped, GradWrapped, VmapWrapped builders
    errors.rs       → ApiError enum wrapping DispatchError with user-friendly messages
```

Dependencies: `fj-api` → `fj-dispatch`, `fj-core`

### 2.2 Public API Signatures

```rust
// transforms.rs

/// JIT-compile a Jaxpr for execution.
pub fn jit(jaxpr: Jaxpr) -> JitWrapped;

/// Reverse-mode gradient of a scalar-to-scalar Jaxpr.
pub fn grad(jaxpr: Jaxpr) -> GradWrapped;

/// Vectorizing map over the leading axis.
pub fn vmap(jaxpr: Jaxpr) -> VmapWrapped;

/// Combined forward eval + gradient in composable form.
pub fn value_and_grad(jaxpr: Jaxpr) -> ValueAndGradWrapped;

// Builder types for deferred execution:
impl JitWrapped {
    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError>;
}

impl GradWrapped {
    pub fn call(&self, args: Vec<Value>) -> Result<(Vec<Value>, Vec<Value>), ApiError>;
}

impl VmapWrapped {
    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError>;
}
```

### 2.3 Error Boundary

```rust
// errors.rs

pub enum ApiError {
    /// Gradient requires scalar first argument and scalar output.
    GradRequiresScalar { detail: String },
    /// Vmap dimension mismatch between mapped arguments.
    VmapDimensionMismatch { expected: usize, actual: usize },
    /// Transform composition is invalid.
    InvalidComposition { detail: String },
    /// Cache key generation failed (strict mode unknown features).
    CacheKeyFailure { detail: String },
    /// Internal evaluation error.
    EvalError { detail: String },
}
```

### 2.4 Retained Crates (no changes)

- `fj-core`: canonical IR, value model, transform model, composition proof
- `fj-dispatch`: dispatch orchestration, transform execution
- `fj-cache`: cache key generation
- `fj-ad`: reverse-mode AD
- `fj-interpreters`: Jaxpr evaluation
- `fj-ledger`: evidence recording

## 3. Implementation Sequence (Unblocks `bd-3dl.13.10`)

1. Create `fj-api` crate with Cargo.toml and workspace registration.
2. Implement `ApiError` wrapping `DispatchError` with user-friendly messages.
3. Implement `jit()` as TTL construction + dispatch wrapper with `Transform::Jit` stack.
4. Implement `grad()` with `Transform::Grad` stack and scalar validation.
5. Implement `vmap()` with `Transform::Vmap` stack and leading-axis semantics.
6. Implement `value_and_grad()` as composition: eval + grad using shared Jaxpr.
7. Add builder combinators: `jit(grad(jaxpr))` via `JitWrapped::compose(GradWrapped)`.
8. Add tests validating all contract rows from `contract_table.v1.json`.

## 4. Risk Register and Controls

| Risk | Impact | Control |
|---|---|---|
| User-facing API diverges from dispatch semantics | silent behavior drift | All `fj-api` functions delegate to `dispatch()` with no local computation |
| Error message leaks internal state | security concern | `ApiError` maps internal errors to user-safe messages; no variable names or memory addresses exposed |
| Transform composition bypass via direct `fj-dispatch` use | inconsistent behavior | `fj-api` is the only user-facing crate; `fj-dispatch` is `pub(crate)` by convention |
| Builder pattern overhead | unnecessary complexity | Builders are thin wrappers; no allocation beyond TTL construction |

## 5. Contract and Security Alignment

This plan is aligned with:
- `artifacts/phase2c/FJ-P2C-002/legacy_anchor_map.v1.json` (27 anchors)
- `artifacts/phase2c/FJ-P2C-002/contract_table.v1.json` (12 strict + 5 hardened rows)
- `artifacts/phase2c/FJ-P2C-002/security_threat_matrix.md` (10 threat classes)

Boundary law:
- Strict mode remains fail-closed for all error paths through `fj-api`.
- Hardened mode divergences are only exposed via explicit `ApiError` variants.
- No `fj-api` function may bypass `verify_transform_composition`.
- All `fj-api` calls produce evidence ledger entries via the dispatch path.

## 6. Definition of Done for `bd-3dl.13.4`

1. `fj-api` crate exists in workspace with compiling skeleton.
2. Public API signatures (`jit`, `grad`, `vmap`, `value_and_grad`) are defined with types.
3. `ApiError` enum is defined with all error variants.
4. Implementation plan document exists with justified architecture decision.
5. Follow-on bead (`bd-3dl.13.10`) has a clear contract-level handoff.
