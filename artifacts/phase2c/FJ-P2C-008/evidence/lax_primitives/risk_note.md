# FJ-P2C-008 Risk Note: LAX Primitive First Wave

## Subsystem

`fj-lax` — LAX primitive evaluation engine. 35 primitive variants (33 active, 2 unsupported).

## Safety Profile

- **Zero unsafe code**: `#![forbid(unsafe_code)]` on the crate
- All arithmetic uses safe Rust f64/i64 operators
- All tensor indexing uses bounds-checked Vec operations
- No raw pointers, transmute, or manual memory management
- Type dispatch via exhaustive Rust enum matching

## Verified Invariants

| # | Invariant | Evidence |
|---|-----------|----------|
| 1 | Each primitive matches JAX semantics for F64/I64/Bool | 160 tests across 3 suites (unit/oracle/e2e) |
| 2 | Type promotion: I64+I64→I64, any-float→F64 | 4 type promotion tests + 11 property tests |
| 3 | Shape inference is deterministic | Pure function of input shapes + params |
| 4 | NaN propagation follows IEEE 754 | 16 NaN/Inf edge case tests |
| 5 | Reduction identity elements are correct | sum→0, prod→1, max→MIN, min→MAX verified |
| 6 | Broadcasting: scalar-tensor only, no rank expansion | 6 broadcasting tests + adversarial shape mismatch |
| 7 | Wrong arity returns ArityMismatch error | 10 error path tests |
| 8 | Integer overflow wraps (2's complement) | Rust i64 wrapping semantics |
| 9 | Transcendental edge cases produce IEEE results | log(0)=-Inf, sqrt(-1)=NaN, exp(-Inf)=0 verified |
| 10 | Reshape incompatible size → actionable error | 3 reshape error tests |

## Known Divergences from JAX

| Item | JAX Behavior | FrankenJAX V1 | Risk |
|------|-------------|---------------|------|
| round(2.5) | Banker's rounding → 2 | Rust f64::round → 3 | Low (documented) |
| Gather/Scatter | Supported | Unsupported error | High (programs using these will fail) |
| broadcast_in_dim | Full NumPy broadcasting | Scalar-tensor only | Medium |
| Multi-axis reduction | Supported | Single-axis only | Low |
| Complex dtypes | complex64/complex128 | Not supported | Medium |

## Performance Summary

| Benchmark | Measured | Target | Status |
|-----------|----------|--------|--------|
| Dispatch overhead | 11ns | <50ns | PASS |
| Add scalar | 11ns | — | — |
| Add 1K i64 vec | 17.6µs | — | — |
| Mul 1K f64 vec | 20.6µs | — | — |
| Dot 100 i64 | 596ns | — | — |
| Reduce sum 1K | 2.2µs | — | — |
| Sin 1K f64 | 25.8µs | — | — |
| Exp 1K f64 | 20.1µs | — | — |
| Reshape 1K | 570ns | — | — |
| Eq 1K i64 | 17.0µs | — | — |

## Recommendation

V1 LAX primitive evaluation is correct and performant. No optimization needed for current workload. Major risk items (Gather/Scatter unsupported, limited broadcasting) are documented and will be addressed in future packets.
