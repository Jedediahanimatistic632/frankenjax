# Dispatch Baseline (2026-02-13)

## Command

```bash
CARGO_TARGET_DIR=/tmp/frankenjax-target cargo bench
```

## Benchmark

- bench id: `dispatch/simple_add`
- criterion time: `[2.5046 us, 2.5202 us, 2.5353 us]`
- samples: `100`
- outliers: `2` (2 mild high)
- criterion change vs previous sample set: `[-4.4025%, -1.7196%, +0.3005%]` (no statistically significant change)

## Scope

This benchmark exercises the current first vertical slice:

- `fj-core` TTL model
- `fj-cache` key derivation
- `fj-interpreters` + `fj-lax` execution
- `fj-dispatch` orchestration
- `fj-ledger` evidence record creation

## Notes

- This is a baseline artifact for the extreme-software-optimization loop.
- Future optimization commits should compare against this benchmark and provide an isomorphism proof summary.
