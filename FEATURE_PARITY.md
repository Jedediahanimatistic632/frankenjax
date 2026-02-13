# FEATURE_PARITY

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Feature Family Matrix

| Feature Family | Status | Current Evidence | Next Required Artifact |
|---|---|---|---|
| Canonical IR + TTL model | parity_green | `fj-core` value/tensor model + transform composition proof tests | Differential trace fixtures from legacy oracle capture |
| Primitive semantics (scalar + tensor core subset) | in_progress | `fj-lax` unit tests for elementwise/broadcast/dot/reduce_sum | Extended oracle-matched primitive corpus beyond current subset |
| Interpreter path over canonical IR | in_progress | `fj-interpreters` scalar + vector tests | Multi-output and higher-rank fixture families |
| Dispatch path + transform wrappers | in_progress | `fj-dispatch` tests for `jit`, `grad`, `vmap`, and order sensitivity | Differential dispatch parity report vs legacy runtime |
| Cache-key determinism + mode split | in_progress | `fj-cache` strict/hardened tests + fail-closed unknown-feature gate | Legacy `cache_key.py` component-by-component parity ledger |
| Decision/evidence ledger foundation | in_progress | `fj-ledger` loss-matrix/action tests + dispatch evidence signals | Calibration/evidence confidence bounds and drift alerts |
| Conformance harness + transform bundle runner | in_progress | `fj-conformance` transform bundle loader/executor/report + passing transform integration tests | Full legacy-oracle captured fixtures (requires `jaxlib` availability) |
| Legacy fixture capture automation | in_progress | `capture_legacy_fixtures.py` script with strict/fallback modes | Strict mode capture run from real `jax` + `jaxlib` environment |
| `jit` transform semantics | in_progress | fixture cases pass in transform bundle (`jit_*`) | Expand to broader API families from `tests/jax_jit_test.py` |
| `grad` transform semantics | in_progress | fixture cases pass in transform bundle (`grad_*`) | Expand to richer autodiff op families from `tests/lax_autodiff_test.py` |
| `vmap` transform semantics | in_progress | fixture cases pass in transform bundle (`vmap_*`) | Expand batching families from `tests/lax_vmap_test.py` |
| RNG determinism | not_started | legacy anchors identified | Differential fixture family from `tests/random_test.py` |
| RaptorQ sidecar durability pipeline | parity_green | sidecar/scrub/decode-proof generation + tests + committed artifacts in `artifacts/durability` | Scale sidecar generation to all long-lived evidence bundles |

## Required Evidence Per Family

1. Differential conformance report.
2. Invariant checklist entry/update.
3. Benchmark delta report for perf-sensitive changes.
4. Risk note update if compatibility/security surface changed.

## Coverage Objective

Target: 100% coverage for scoped compatibility families (feature-complete for declared V1 scope), with explicit parity exceptions documented and justified.
