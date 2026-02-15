# FJ-P2C-003 Security + Compatibility Threat Matrix (bd-3dl.14.3)

## Scope

Packet boundary: Partial evaluation and staging subsystem.

Primary subsystems:
- `fj-interpreters` (partial eval core)
- `fj-trace` (dynamic tracing)
- `fj-core` (PartialVal, Jaxpr manipulation)
- `fj-dispatch` (staging pipeline integration)

## Threat Matrix

| Threat class | Attack vector | Strict mitigation | Hardened mitigation | Fail-closed boundary | Residual risk | Evidence |
|---|---|---|---|---|---|---|
| Residual graph inflation | Adversarial input where no equations can be folded, producing a residual Jaxpr as large as the original with additional residual variables | Residual size bounded by original Jaxpr size plus residual count; no amplification beyond O(n) where n = original equation count | Equation count pressure guard (inv009 from P2C-001) applies to residual; staging timeout (5s) aborts runaway PE | Residual Jaxpr passes same well-formedness checks as original | Medium: no explicit residual size cap enforced yet | `artifacts/phase2c/FJ-P2C-003/contract_table.v1.json` |
| Constant hoisting confusion | Value classified as known at trace-time (via PartialVal::Known) changes semantics at runtime due to mutation or aliasing | Constants are cloned at hoisting time; identity-based dedup prevents aliasing; Rust ownership model prevents mutation after hoisting | Same as strict; additionally audit-logs any constant that fails value comparison at dispatch time | Constant value mismatch blocks execution with reason_code `pe_constant_duplication` | Low: Rust's ownership model prevents the primary mutation vector | `artifacts/phase2c/FJ-P2C-003/legacy_anchor_map.v1.json` |
| Staging loop via recursive make_jaxpr | Recursive calls to make_jaxpr() or nested jit causing stack overflow or unbounded trace depth | Stack depth limit enforced by Rust's default stack size; recursive jit within strict mode fails closed | Recursive jit collapsed to single staging pass (inv010); nesting depth logged | Stack overflow caught by Rust runtime; no undefined behavior possible due to `#![forbid(unsafe_code)]` | Low: Rust stack guard prevents true overflow | `artifacts/phase2c/FJ-P2C-003/contract_table.v1.json` |
| Timing side-channel via partial eval | Partial evaluation time reveals information about which inputs are known vs unknown, potentially leaking constant values | Not mitigated in V1; PE is inherently data-dependent in timing | Not mitigated; classified as accepted risk for V1 scope | N/A - informational threat only | High: timing variation is fundamental to PE; constant-time PE would eliminate the optimization benefit | Acknowledged risk |
| Tracer escape/leak | Tracer objects from tracing context escape into returned values or are captured by closures, causing undefined tracing behavior | ensure_no_leaks guard fires before Jaxpr finalization (inv008); escaped tracers produce immediate error | Same as strict; no repair path for leaked tracers | Leaked tracer detection is fail-closed in both modes | Low: Rust's type system provides additional static guarantees beyond Python's runtime checks | `artifacts/phase2c/FJ-P2C-003/legacy_anchor_map.v1.json` |
| DCE bypass via fake effects | Crafted equation claims side effects to prevent DCE removal, bloating the residual Jaxpr | Effect validation at equation creation time; only recognized effect types are accepted | Same as strict; unrecognized effects are rejected | Unknown effect types fail closed | Low: effect enum is closed in FrankenJAX (no arbitrary effect registration) | `artifacts/phase2c/FJ-P2C-003/contract_table.v1.json` |
| Malformed PartialVal injection | PartialVal constructed with both known and unknown fields set (violating XOR invariant) | Type-level enforcement via Rust enum (Known/Unknown variants make invalid states unrepresentable) | Hardened mode adds runtime assertion for defense-in-depth (inv012) | Invalid PartialVal state is compile-time impossible in Rust enum representation | Negligible: Rust's type system eliminates this threat class entirely | `crates/fj-core/src/lib.rs` (planned) |

## Compatibility Envelope

| JAX staging behavior | FrankenJAX status | strict mode | hardened mode | evidence |
|---|---|---|---|---|
| make_jaxpr() basic functionality | SUPPORTED (scalar/vector) | guaranteed | guaranteed | `artifacts/phase2c/FJ-P2C-003/legacy_anchor_map.v1.json` |
| trace_to_jaxpr_dynamic (abstract tracing) | SUPPORTED (scalar/vector, 37 primitives) | guaranteed | guaranteed | `crates/fj-trace/src/lib.rs` |
| partial_eval_jaxpr_nounits (Jaxpr splitting) | PLANNED | guaranteed once implemented | guaranteed with hardened fallbacks | `artifacts/phase2c/FJ-P2C-003/contract_table.v1.json` |
| Staged compilation pipeline (Traced->Lowered->Compiled) | PARTIAL (CPU eval only, no XLA/MLIR lowering) | CPU eval path guaranteed | CPU eval path guaranteed | `crates/fj-interpreters/src/lib.rs` |
| Dead code elimination (dce_jaxpr) | PLANNED | guaranteed once implemented | guaranteed | `artifacts/phase2c/FJ-P2C-003/legacy_anchor_map.v1.json` |
| Higher-order primitives (cond, while_loop, scan) | NOT in V1 scope | out-of-scope | out-of-scope | N/A |
| Dynamic shapes / AbstractedAxisSize | NOT in V1 scope | out-of-scope | out-of-scope | N/A |
| Residual Jaxpr format | COMPATIBLE (same Jaxpr type as traced) | guaranteed | guaranteed | `crates/fj-core/src/lib.rs` |

## Explicit Fail-Closed Rules

1. Residual type mismatches between jaxpr_known outputs and jaxpr_unknown inputs terminate the request.
2. PartialVal XOR invariant violations are compile-time impossible (Rust enum); runtime assertions added for defense-in-depth in hardened mode.
3. Tracer leaks are detected and rejected before any Jaxpr is returned from tracing.
4. DCE never removes equations with non-dceable effects; unrecognized effect types are rejected.
5. Staging timeouts (hardened mode) abort cleanly with no partial state leakage.
6. Constants are cloned at hoisting time; no aliasing or mutation vectors exist post-hoisting.
