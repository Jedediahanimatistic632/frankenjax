# PROPOSED_ARCHITECTURE

## 1. Architecture Constraints

1. Spec-first implementation from explicit behavior docs.
2. Strict/hardened mode split at compatibility boundaries.
3. Fail-closed on unknown incompatible features in strict mode.
4. Profile-first optimization with behavior-isomorphism proofs.
5. RaptorQ sidecar durability for long-lived evidence artifacts.

## 2. Current Workspace Crate Map

- `fj-core`: canonical data model, tensor value model, Trace Transform Ledger (TTL), transform composition proofs.
- `fj-lax`: scoped primitive execution semantics (scalar + tensor subset).
- `fj-interpreters`: interpreter path over canonical IR and runtime values.
- `fj-cache`: deterministic cache-key derivation + compatibility gates.
- `fj-ledger`: decision/evidence ledger artifacts.
- `fj-dispatch`: dispatch pipeline integrating cache/interpreter/ledger plus transform wrappers (`jit`/`grad`/`vmap`).
- `fj-runtime`: runtime admission model and integration bridges.
- `fj-conformance`: fixture manifests, transform parity runner, and durability pipeline.

## 3. End-to-End Dataflow (Current Slice)

`request` -> `TTL proof` -> `cache key` -> `transform wrappers` -> `interpreter` -> `decision ledger` -> `response`

## 4. Target Dataflow (Full)

`user API` -> `trace` -> `canonical IR` -> `transform stack` -> `lowering` -> `runtime backend`

## 5. Strict vs Hardened Mode Contract

- Strict mode:
  - maximize compatibility
  - reject unknown incompatible metadata
  - no behavior-changing repairs
- Hardened mode:
  - preserve outward contract
  - allow bounded defensive handling
  - log compatibility events to evidence ledger

## 6. Asupersync Integration Contract

Runtime integration points include:

- cancellation-aware checkpoints during long-running compile/dispatch operations
- budget/deadline propagation for controlled degradation
- deterministic outcome capture for policy decisions and auditability

## 7. FrankenTUI Integration Contract

Operational UI integration points include:

- real-time display of transform stack and cache decision state
- progressive disclosure of evidence ledger details
- parity/benchmark status dashboards for operator workflows

## 8. Conformance and Benchmark Architecture

Conformance:

- fixture bundle (`transform-fixtures.v1`) -> case runner -> `fj-dispatch` execution -> tolerance comparison -> parity report

Benchmark:

- criterion benchmark (`dispatch/simple_add`) tracked in `artifacts/performance/dispatch_baseline_2026-02-13.md`
- warm/cold cache and transform overhead deltas to be expanded per family

## 9. Durability Architecture (RaptorQ-Everywhere)

Implemented durability module: `fj-conformance::durability`.

Pipeline:

1. Encode artifact bytes to RaptorQ symbols via Asupersync `EncodingPipeline`.
2. Persist sidecar manifest with symbol payloads.
3. Scrub by decoding and comparing SHA-256 to original bytes.
4. Emit decode proof artifacts under simulated symbol loss.

Current committed durability artifacts:

- `artifacts/durability/legacy_transform_cases.v1.sidecar.json`
- `artifacts/durability/legacy_transform_cases.v1.scrub.json`
- `artifacts/durability/legacy_transform_cases.v1.decode-proof.json`
- `artifacts/durability/dispatch_baseline_2026-02-13.sidecar.json`
- `artifacts/durability/dispatch_baseline_2026-02-13.scrub.json`
- `artifacts/durability/dispatch_baseline_2026-02-13.decode-proof.json`

Operational CLI:

- `cargo run -p fj-conformance --bin fj_durability -- generate ...`
- `cargo run -p fj-conformance --bin fj_durability -- scrub ...`
- `cargo run -p fj-conformance --bin fj_durability -- proof ...`
- `cargo run -p fj-conformance --bin fj_durability -- pipeline ...`

## 10. Evolution Plan

- Preserve crate boundaries while expanding primitive/transform coverage.
- Add legacy-oracle strict capture runs once `jaxlib` environment is available.
- Expand drift-gate compatibility matrix to cover cache-key and transform semantics at subsystem granularity.
- Add lowering/backend crates once conformance coverage for current slice is stable.
