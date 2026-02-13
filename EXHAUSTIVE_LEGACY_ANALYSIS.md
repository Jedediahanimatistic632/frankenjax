# EXHAUSTIVE_LEGACY_ANALYSIS.md — FrankenJAX

Date: 2026-02-13  
Method stack: `$porting-to-rust` Phase-2 Deep Extraction + `$alien-artifact-coding` + `$extreme-software-optimization` + RaptorQ durability + frankenlibc/frankenfs strict/hardened doctrine.

## 0. Mission and Completion Criteria

This document defines exhaustive legacy extraction for FrankenJAX. Phase-2 is complete only when each scoped subsystem has:
1. explicit invariants,
2. explicit crate ownership,
3. explicit oracle families,
4. explicit strict/hardened policy behavior,
5. explicit performance and durability gates.

## 1. Source-of-Truth Crosswalk

Legacy corpus:
- `/data/projects/frankenjax/legacy_jax_code/jax`
- Upstream oracle: `jax-ml/jax`

Project contracts:
- `/data/projects/frankenjax/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md`
- `/data/projects/frankenjax/EXISTING_JAX_STRUCTURE.md`
- `/data/projects/frankenjax/PLAN_TO_PORT_JAX_TO_RUST.md`
- `/data/projects/frankenjax/PROPOSED_ARCHITECTURE.md`
- `/data/projects/frankenjax/FEATURE_PARITY.md`

Important specification gap:
- the comprehensive spec currently defines sections `0-13` and then jumps to `21`; detailed sections for crate contracts/conformance matrix/threat matrix/perf budgets/CI/RaptorQ envelope are missing and must be backfilled before release governance is trustworthy.

## 2. Quantitative Legacy Inventory (Measured)

- Total files: `1830`
- Python: `1005`
- Native: `cc=123`, `h=90`, `c=2`, `cu=1`
- Test-like files: `312`

High-density zones:
- `jax/_src/pallas` (69 files)
- `jax/experimental/jax2tf` (61)
- `jax/_src/internal_test_util` (52)
- `jax/experimental/pallas` (50)
- `jax/_src/scipy` (47)

## 3. Subsystem Extraction Matrix (Legacy -> Rust)

| Legacy locus | Non-negotiable behavior to preserve | Target crates | Primary oracles | Phase-2 extraction deliverables |
|---|---|---|---|---|
| `jax/_src/core.py` | `Trace`/`Tracer`/`Jaxpr` typing and construction invariants | `fj-core` | `tests/jaxpr_util_test.py`, `tests/jaxpr_effects_test.py` | IR schema and typing law ledger |
| `jax/_src/interpreters/partial_eval.py` | trace-to-jaxpr construction, residual and leak constraints | `fj-interpreters` | `tests/api_test.py`, `tests/extend_test.py` | partial-eval state machine + residual contract |
| `jax/interpreters/{ad,batching,pxla,mlir}.py` | transform composition semantics | `fj-interpreters`, `fj-dispatch` | transform suites in `tests/*` | composition equivalence matrix |
| `jax/_src/api.py` | jit/grad/vmap observable API contracts | `fj-dispatch` | `tests/api_test.py` | API decision table and error-surface map |
| `jax/_src/lax/*` | primitive semantics under transforms | `fj-lax` | `tests/lax_test.py`, `tests/lax_numpy_test.py` | primitive contract corpus |
| `jax/_src/dispatch.py` | runtime token/effect sequencing | `fj-dispatch`, `fj-runtime` | `tests/xla_interpreter_test.py` | effect-token sequencing ledger |
| `jax/_src/compilation_cache.py`, `lru_cache.py` | cache keying and lifecycle determinism | `fj-cache` | `tests/compilation_cache_test.py`, `tests/cache_key_test.py` | cache-key schema + invalidation rules |
| `jax/_src/xla_bridge.py` + `jaxlib/*.cc` | backend selection, FFI bridge correctness | `fj-runtime` | `tests/xla_bridge_test.py` | backend/ffi boundary register |

## 4. Alien-Artifact Invariant Ledger (Formal Obligations)

- `FJ-I1` Jaxpr typing integrity: every produced IR graph remains well-typed under scoped transforms.
- `FJ-I2` Trace leak prohibition: no escaped tracer across transform boundaries.
- `FJ-I3` Composition determinism: scoped transform compositions are order-consistent where contract requires.
- `FJ-I4` Cache key soundness: semantically equivalent inputs map consistently; non-equivalent inputs do not collide silently.
- `FJ-I5` Effect sequencing safety: runtime token ordering preserves observable side-effect semantics.

Required proof artifacts per implemented slice:
1. invariant statement,
2. executable witness fixtures,
3. counterexample archive,
4. remediation proof.

## 5. Native/XLA/FFI Boundary Register

| Boundary | Files | Risk | Mandatory mitigation |
|---|---|---|---|
| backend bridge | `jax/_src/xla_bridge.py` | critical | backend selection fixture matrix |
| ffi registration/calls | `jax/_src/ffi.py`, `jaxlib/ffi.*` | high | callback lifetime and registration-order tests |
| runtime client/device/program | `jaxlib/py_client*.cc`, `py_device*.cc`, `py_program.cc` | critical | ownership/lifetime stress fixtures |
| host callbacks and transfers | `py_host_callback.cc`, `py_socket_transfer.cc` | high | async transfer and callback race corpus |

## 6. Compatibility and Security Doctrine (Mode-Split)

Decision law (runtime):
`mode + trace_contract + risk_score + budget -> allow | full_validate | fail_closed`

| Threat | Strict mode | Hardened mode | Required ledger artifact |
|---|---|---|---|
| malformed trace graph | fail-closed | fail-closed with bounded diagnostics | trace incident ledger |
| cache poisoning/collision risk | strict key checks | stricter admission and audit | cache integrity report |
| backend confusion | fail unknown backend/protocol | fail unknown backend/protocol | backend decision ledger |
| callback lifetime hazard | fail invalid lifecycle state | quarantine and fail with trace | ffi lifecycle report |
| unknown incompatible runtime metadata | fail-closed | fail-closed | compatibility drift report |

## 7. Conformance Program (Exhaustive First Wave)

### 7.1 Fixture families

1. API transform fixtures (`jit`, `grad`, `vmap`)
2. Jaxpr and effects fixtures
3. primitive/lax numerical fixtures
4. cache key and compilation cache fixtures
5. RNG/state fixtures
6. sharding and multi-device fixtures

### 7.2 Differential harness outputs (`fj-conformance`)

Each run emits:
- machine-readable parity report,
- mismatch class histogram,
- minimized repro fixture bundle,
- strict/hardened divergence report.

Release gate rule: critical-family drift => hard fail.

## 8. Extreme Optimization Program

Primary hotspots:
- trace-to-jaxpr path
- transform composition path
- dispatch/lowering path
- cache lookup and serialization path

Current governance state:
- comprehensive spec now includes sections 14-20 with explicit budgets and gate topology; next step is empirical calibration.

Provisional Phase-2 budgets (must be ratified into spec):
- transform composition overhead regression <= +10%
- cache hit path p95 regression <= +8%
- p99 regression <= +10%, peak RSS regression <= +10%

Optimization governance:
1. baseline,
2. profile,
3. one lever,
4. conformance proof,
5. budget gate,
6. evidence commit.

## 9. RaptorQ-Everywhere Artifact Contract

Durable artifacts requiring RaptorQ sidecars:
- conformance fixture bundles,
- benchmark baselines,
- cache-key schema ledgers,
- risk/proof ledgers.

Required envelope fields:
- source hash,
- symbol manifest,
- scrub status,
- decode proof chain.

## 10. Phase-2 Execution Backlog (Concrete)

1. Extract IR typing rules from `core.py`.
2. Extract partial-eval residual and leak constraints.
3. Extract AD/batching/pxla composition semantics.
4. Extract API argument normalization and error contracts.
5. Extract lax primitive semantic contracts for scoped subset.
6. Extract dispatch token sequencing rules.
7. Extract cache-key schema and lifecycle behavior.
8. Extract backend bridge and FFI lifecycle rules.
9. Build first differential fixture corpus for items 1-8.
10. Implement mismatch taxonomy in `fj-conformance`.
11. Add strict/hardened divergence reporting.
12. Add RaptorQ sidecar generation and decode-proof validation.
13. Ratify section-14-20 budgets/gates against first benchmark and conformance runs.

Definition of done for Phase-2:
- each section-3 row has extraction artifacts,
- all six fixture families runnable,
- governance sections 14-20 are empirically ratified and tied to harness outputs.

## 11. Residual Gaps and Risks

- sections 14-20 now exist; top non-code risk is uncalibrated budget thresholds until first benchmark cycle lands.
- `PROPOSED_ARCHITECTURE.md` crate map formatting has literal `\n`; normalize before automation.
- backend and FFI boundaries remain highest regression risk until corpus breadth increases.

## 12. Deep-Pass Hotspot Inventory (Measured)

Measured from `/data/projects/frankenjax/legacy_jax_code/jax`:
- file count: `1830`
- concentration: `jax/_src` (`371` files), `jax/experimental` (`214`), plus broad test and backend surfaces

Top source hotspots by line count (first-wave extraction anchors):
1. `tests/pjit_test.py` (`11166`)
2. `jax/_src/numpy/lax_numpy.py` (`9645`)
3. `jax/_src/lax/lax.py` (`9107`)
4. `tests/api_test.py` (`7983`)
5. `tests/lax_numpy_test.py` (`6531`)
6. `jax/_src/pallas/mosaic/lowering.py` (`4460`)

Interpretation:
- JAX behavior is defined jointly by tracing internals and broad test contracts,
- IR/transforms/dispatch/cache boundaries need strict extraction discipline,
- backend and FFI contracts remain highest operational risk.

## 13. Phase-2C Extraction Payload Contract (Per Ticket)

Each `FJ-P2C-*` ticket MUST produce:
1. IR/type/state inventory (jaxpr/tracer/effect structures),
2. transform decision tables (`jit`/`grad`/`vmap` paths),
3. cache and backend routing rule ledger,
4. error and diagnostics contract map,
5. strict/hardened mode split policy,
6. explicit exclusions,
7. fixture mapping manifest,
8. optimization candidate + isomorphism risk note,
9. RaptorQ artifact declaration,
10. compatibility backfill notes for comprehensive-spec governance sections.

Artifact location (normative):
- `artifacts/phase2c/FJ-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FJ-P2C-00X/contract_table.md`
- `artifacts/phase2c/FJ-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FJ-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FJ-P2C-00X/risk_note.md`

## 14. Strict/Hardened Compatibility Drift Budgets

Packet acceptance budgets:
- strict critical drift budget: `0`
- strict non-critical drift budget: `<= 0.10%`
- hardened divergence budget: `<= 1.00%` and allowlisted only
- unknown backend/cache/ffi metadata: fail-closed

Per-packet report requirements:
- `strict_parity`,
- `hardened_parity`,
- `transform_drift_summary`,
- `backend_route_drift_summary`,
- `compatibility_drift_hash`.

## 15. Extreme-Software-Optimization Execution Law

Mandatory loop:
1. baseline,
2. profile,
3. one lever,
4. conformance + invariant replay,
5. re-baseline.

Primary sentinel workloads:
- transform composition traces (`FJ-P2C-001..003`),
- dispatch/token sequencing (`FJ-P2C-004`),
- cache-key churn workloads (`FJ-P2C-005`),
- lax primitive throughput tests (`FJ-P2C-008`).

Optimization scoring gate:
`score = (impact * confidence) / effort`, merge only if `score >= 2.0`.

## 16. RaptorQ Evidence Topology and Recovery Drills

Durable artifacts requiring sidecars:
- parity reports,
- transform mismatch corpora,
- backend/cache compatibility ledgers,
- benchmark baselines,
- strict/hardened decision logs.

Naming convention:
- payload: `packet_<id>_<artifact>.json`
- sidecar: `packet_<id>_<artifact>.raptorq.json`
- proof: `packet_<id>_<artifact>.decode_proof.json`

Decode-proof failures are release blockers.

## 17. Phase-2C Exit Checklist (Operational)

Phase-2C is complete only when:
1. `FJ-P2C-001..008` artifact packs exist and pass validation.
2. All packets have strict and hardened fixture coverage.
3. Drift budgets from section 14 are satisfied.
4. High-risk packets include optimization proof artifacts.
5. RaptorQ sidecars + decode proofs are scrub-clean.
6. Governance backfill tasks are explicitly tied to packet outputs.
