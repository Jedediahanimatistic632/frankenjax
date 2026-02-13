# FrankenJAX

<div align="center">
  <img src="frankenjax_illustration.webp" alt="FrankenJAX - Clean-room Rust reimplementation of JAX transform semantics">
</div>

FrankenJAX is a clean-room Rust reimplementation targeting semantic fidelity, mathematical rigor, operational safety, and profile-proven performance.

## Core Identity

Trace Transform Ledger (TTL): canonical JAXPR-like IR with transform-composition evidence for `jit`, `grad`, and `vmap`.

Transform composition semantics are non-negotiable.

## Method Stack

This project applies four disciplines for meaningful changes:

1. `alien-artifact-coding`
2. `extreme-software-optimization`
3. `RaptorQ-everywhere`
4. strict/hardened compatibility-security mode split

## Legacy Oracle

- `/dp/frankenjax/legacy_jax_code/jax`

## Exemplar Reference

- `references/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`

## Workspace Crates

- `fj-core`
- `fj-lax`
- `fj-interpreters`
- `fj-cache`
- `fj-ledger`
- `fj-dispatch`
- `fj-runtime`
- `fj-conformance`

## Current Status

Implemented foundation + differential harness slice:

- canonical IR + tensor-aware runtime value model
- transform composition proof checks and order-sensitive dispatch execution
- scoped primitive interpreter path (`add`, `mul`, `dot`, `sin`, `cos`, `reduce_sum` subset)
- deterministic cache-key module with strict/hardened gate behavior
- decision/evidence ledger primitives with loss-matrix actions
- transform fixture bundle runner for `jit`/`grad`/`vmap`
- RaptorQ sidecar + scrub + decode-proof pipeline for long-lived artifacts

## Fixture Capture

Regenerate transform fixtures:

```bash
python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json
```

Notes:

- strict legacy capture requires compatible `jax` + `jaxlib`
- when unavailable, script falls back to deterministic analytical capture unless `--strict` is set

## Durability Commands

Generate sidecar only:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  generate --artifact <artifact_path> --sidecar <sidecar_path>
```

Scrub sidecar:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  scrub --artifact <artifact_path> --sidecar <sidecar_path> --report <scrub_report_path>
```

Generate decode proof:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  proof --artifact <artifact_path> --sidecar <sidecar_path> --proof <decode_proof_path> --drop-source 2
```

All-in-one pipeline:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  pipeline --artifact <artifact_path> --sidecar <sidecar_path> --report <scrub_report_path> --proof <decode_proof_path>
```

## Key Documents

- `AGENTS.md`
- `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md`
- `PLAN_TO_PORT_JAX_TO_RUST.md`
- `EXISTING_JAX_STRUCTURE.md`
- `PROPOSED_ARCHITECTURE.md`
- `FEATURE_PARITY.md`

## Verification Commands

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo test -p fj-conformance -- --nocapture
cargo bench
```
