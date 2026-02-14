# FrankenJAX Test Conventions (v1)

## Naming Rules

- Unit test: `test_<subsystem>_<behavior>_<condition>`
- Property test: `prop_<subsystem>_<invariant>`
- E2E test: `e2e_<scenario_id>`

Examples:

- `test_cache_key_unknown_features_fail_closed`
- `prop_dispatch_transform_order_deterministic`
- `e2e_transform_fixture_bundle_smoke`

## Structured Log Contract

All new tests should emit or construct a `frankenjax.test-log.v1` record through `fj-test-utils::TestLogV1`.

Required identity fields:

- `test_id`
- `fixture_id` (SHA-256 over canonical fixture JSON)
- `mode`
- `result`
- `seed` (when property test seed exists)

## Property Test Configuration

Use `fj_test_utils::property_test_case_count()` for test-case volume:

- local default: `256`
- CI default: `1024`
- override: `FJ_PROPTEST_CASES=<N>`

Seed capture precedence:

1. `FJ_PROPTEST_SEED`
2. `PROPTEST_RNG_SEED`

## Coverage Gate Policy

Coverage should be measured with `cargo-llvm-cov` and compared against per-crate floors.

Suggested command:

```bash
cargo llvm-cov --workspace --lcov --output-path artifacts/testing/coverage.lcov
```

Initial floor targets:

- core crates (`fj-core`, `fj-dispatch`, `fj-lax`, `fj-cache`): line >= 90%, branch >= 80%
- supporting crates (`fj-ad`, `fj-interpreters`, `fj-runtime`, `fj-ledger`, `fj-egraph`, `fj-conformance`): line >= 85%, branch >= 75%

