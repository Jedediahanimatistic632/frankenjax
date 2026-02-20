# EXHAUSTIVE_LEGACY_ANALYSIS.md — FrankenJAX

Date: 2026-02-13  
Method stack: `$porting-to-rust` Phase-2 Deep Extraction + `$alien-artifact-coding` + `$extreme-software-optimization` + RaptorQ durability + frankenlibc/frankenfs strict/hardened doctrine.
DOC-PASS-00 baseline matrix: `artifacts/docs/bd-3dl.23.1_gap_matrix.v1.md`

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

Source-anchored foundation artifacts for this doctrine:
- `artifacts/phase2c/FJ-P2C-FOUNDATION/security_threat_matrix.md`
- `artifacts/phase2c/FJ-P2C-FOUNDATION/fail_closed_audit.md`
- `artifacts/phase2c/global/compatibility_matrix.v1.json`
- `artifacts/phase2c/FJ-P2C-FOUNDATION/risk_note.v1.json`

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

## 18. Data Model, State, and Invariant Mapping (DOC-PASS-03)

### 18.1 Canonical Data Model Inventory

| Model entity | Primary owner | Core state fields | Non-negotiable invariants | Violation handling |
|---|---|---|---|---|
| Trace graph (`Trace`/`Tracer`/`Jaxpr`) | legacy: `jax/_src/core.py`; Rust: `fj-core` | var ids, equations, in/out vars, transform stack | every referenced var is defined; outputs do not shadow prior vars; canonical fingerprint determinism | strict: fail-closed with typed error; hardened: emit ledger event + bounded recovery only where contract allows |
| Transform composition proof (`jit`/`grad`/`vmap`) | legacy: interpreter stack; Rust: `fj-core` + `fj-dispatch` | ordered transform vector, per-transform evidence | supported sequence only; evidence count equals transform count; composition signature stable | strict: reject unsupported composition; hardened: allow only documented bounded fallback and log divergence |
| Cache identity (`cache_key`) | legacy: `cache_key.py`; Rust: `fj-cache` | backend id/version, compile options, flags, accelerator config, hooks | semantically equivalent inputs hash consistently; unknown incompatible metadata is not silently accepted in strict mode | strict: fail-closed on unknown incompatible metadata; hardened: explicit compatibility event + allowlisted continuation |
| Dispatch request/response envelope | legacy: `dispatch.py`; Rust: `fj-dispatch` | mode, backend, transforms, args, options, unknown feature list | transform order preserved; request immutability during execution; response deterministic for deterministic inputs | strict: surface deterministic error; hardened: bounded fallback paths with audit signal |
| Evidence/decision ledger | legacy analog spread across runtime policies; Rust: `fj-ledger` | evidence signals, posterior, loss matrix, action record | append-only ordering; calibrated posterior computation; decision traceability | strict/hardened: never drop records; always emit deterministic decision metadata |
| Durability sidecar artifacts | legacy N/A (new contract); Rust: `fj-conformance::durability` | symbol manifest, source hash, scrub/proof reports | scrub hash equals source hash; decode proof succeeds for configured symbol loss | strict/hardened: decode-proof failure is release blocker (no silent bypass) |

### 18.2 Critical State Transitions

| Subsystem | States | Valid transitions | Invalid transition example | Required recovery/audit output |
|---|---|---|---|---|
| Trace/Jaxpr construction | `Init -> CollectEq -> SealOutvars -> Fingerprint` | monotonic equation append, then seal, then fingerprint | fingerprint before outvar seal | emit invariant failure with offending var/equation and stop in strict mode |
| Transform execution stack | `Request -> Verify -> Apply(jit/grad/vmap)* -> Eval -> Return` | each transform consumes one stack entry in declared order | skipping verify and executing transforms directly | write transform-order incident to ledger; fail closed in strict mode |
| Cache lifecycle | `CanonicalizeInput -> Hash -> Lookup -> Hit|Miss -> (Compile -> Write)` | read-before-write and deterministic key generation | write cache entry without canonical key | cache-integrity incident + block write path |
| Runtime admission | `CollectEvidence -> PosteriorEstimate -> RecommendAction -> Keep|Kill|Reprofile` | decision derived from current posterior and loss matrix | emitting action without posterior/evidence update | log policy inconsistency and deny unsafe action |
| Durability pipeline | `GenerateSidecar -> Scrub -> DecodeProof -> Gate` | scrub and proof must bind to same artifact hash | proof generated against stale sidecar | mark artifact invalid and block release path |

### 18.3 Invariant Violation and Recovery Semantics

| Invariant band | Detector | Strict mode behavior | Hardened mode behavior | Evidence artifact |
|---|---|---|---|---|
| Trace/Jaxpr shape or var integrity | IR validator + interpreter checks | immediate fail-closed | bounded diagnostic + fail-closed unless explicitly allowlisted | parity report + risk note |
| Transform composition mismatch | composition proof verifier | reject request | bounded fallback only if contractually defined; always log | transform drift summary |
| Cache metadata incompatibility | cache-key compatibility gate | reject key generation | allow only audited compatibility path with explicit event | compatibility matrix row + ledger event |
| Dispatch/order anomaly | dispatch invariant checks | stop execution | stop or bounded fallback with mandatory audit trace | decision/evidence ledger |
| Artifact durability mismatch | scrub/proof verifier | block release | block release | sidecar/scrub/decode-proof triplet |

## 19. Execution-Path Tracing and Control-Flow Narratives (DOC-PASS-04)

### 19.1 End-to-End Path E1: Request -> Trace -> Transform -> Runtime

Nominal path:

1. API layer receives transformed program intent (`jit`/`grad`/`vmap` family).
2. Trace/Jaxpr model is built and normalized into canonical IR.
3. Transform composition proof is validated before execution.
4. Compatibility-gated cache key is derived.
5. Dispatch applies transform wrappers in stack order.
6. Interpreter evaluates residual IR path and returns outputs.
7. Evidence ledger and runtime admission model emit decision metadata.

Mandatory branch handling:

| Branch | Trigger | Strict mode | Hardened mode |
|---|---|---|---|
| transform-order branch | unsupported sequence / evidence count mismatch | fail-closed before execution | bounded fallback only if contract allows + audit record |
| cache-compat branch | unknown incompatible metadata | fail-closed key path | compatibility event + bounded continuation |
| interpreter-integrity branch | malformed var/shape/graph refs | deterministic error and stop | deterministic error and stop (with additional diagnostics) |
| policy branch | low-confidence admission posterior | choose conservative action (`Kill`/`Reprofile`) | same action with richer policy trace |

### 19.2 End-to-End Path E2: Cache Hit/Miss and Compilation Decisioning

1. Canonical key material assembled.
2. Lookup branch:
   - hit: short-circuit execution path.
   - miss: full evaluation/compile path.
3. Miss branch completes with guarded cache write.

Safety-critical side branches:
- stale/corrupt cache read -> bypass and record integrity incident.
- unstable/untrusted key input -> refuse write to persistent cache.
- backend mismatch -> fail-closed (no speculative reroute).

### 19.3 End-to-End Path E3: Differential Conformance Pipeline

1. Fixture corpus loaded by family and mode.
2. Cases executed through dispatch path.
3. Comparator branch classifies output (`exact`/`approx`/`shape`/`type`).
4. Drift report emitted with strict/hardened split.
5. Failure branch emits replayable forensic bundle.

Non-negotiable branch law:
- comparator mismatch may never be downgraded silently,
- strict critical drift budget remains zero,
- hardened divergence must be allowlisted and auditable.

### 19.4 End-to-End Path E4: Durability Scrub/Proof Gate

1. Generate sidecar repair symbols.
2. Scrub integrity against source hash.
3. Execute decode proof under configured symbol loss.
4. Gate release on pass/pass outcome.

Failure choreography:
- scrub mismatch -> invalidate artifact, block release.
- decode-proof failure -> block release and require regenerated evidence.
- stale sidecar/proof pairing -> fail-closed until artifact tuple is re-bound.

### 19.5 Verification Crosswalk

Execution-path sections in this document are explicitly tied to:
- `bd-3dl.12.5` (unit/property + logging coverage),
- `bd-3dl.12.6` (differential/metamorphic/adversarial checks),
- `bd-3dl.12.7` (E2E replay/forensics),
- `bd-3dl.23.10` (docs-to-test/logging crosswalk),
- `bd-3dl.23.11` and `bd-3dl.23.12` (integrated draft passes).

## 20. Complexity, Performance, and Memory Characterization (DOC-PASS-05)

### 20.1 Legacy Algorithmic Complexity Classes

| Legacy subsystem | Primary function | Complexity class | Key dimension | Code anchor |
|---|---|---|---|---|
| Jaxpr evaluation | `core.py:eval_jaxpr` | O(E) | E = equations | `jax/_src/core.py:729-754` |
| Dead variable cleanup | `core.py:clean_up_dead_vars` | O(k) per equation | k = dead vars per step | `jax/_src/core.py:4002-4008` |
| Last-used analysis | `core.py:last_used` | O(E * K) | K = avg equation arity | `jax/_src/core.py:3992-4000` |
| Tracer leak detection | `core.py` gc scan | O(heap) | gc-reachable objects | `jax/_src/core.py:1495-1569` |
| Partial-eval trace | `partial_eval.py:trace_to_jaxpr` | O(ops) | ops = traced operations | `jax/_src/interpreters/partial_eval.py:525-596` |
| Partial-eval call split | `partial_eval.py:process_call` | O(n * m) | n = inputs, m = nesting | `jax/_src/interpreters/partial_eval.py:216-270` |
| Dead code elimination | `partial_eval.py:dce_jaxpr` | O(E * K) | E = equations, K = outputs | `jax/_src/interpreters/partial_eval.py:1411-1510` |
| AD linearization | `ad.py:linearize_jaxpr` | O(ops) | traced operations | `jax/_src/interpreters/ad.py:148-224` |
| Batch processing | `batching.py:process_primitive` | O(K) per primitive | K = input count | `jax/_src/interpreters/batching.py:244-250` |
| Cache key derivation | `cache_key.py:get` | O(S) | S = serialized IR size | `jax/_src/cache_key.py:75-149` |
| Cache key deepcopy | `cache_key.py:_hash_serialized_compile_options` | O(C) | C = CompileOptions size | `jax/_src/cache_key.py:277-331` |
| Cache executable I/O | `compilation_cache.py` get/put | O(X) | X = executable size | `jax/_src/compilation_cache.py:276-351` |
| IR tree walk | `compiler.py:_walk_operations` | O(N) | N = IR nodes | `jax/_src/compiler.py:80-90` |
| Dispatch primitive | `dispatch.py:apply_primitive` | O(1) cached | per-primitive cache | `jax/_src/dispatch.py:84-94` |

### 20.2 Legacy Memory Growth Patterns

| Growth driver | Component | Growth formula | Peak trigger | Mitigation in legacy |
|---|---|---|---|---|
| Tracer accumulation | `partial_eval.py`, `core.py` | O(T) where T = total tracers | complex nested traces | dead variable cleanup during eval |
| Residual forwarding | `partial_eval.py:process_call` | O(R * D) where R = residuals, D = call nesting depth | nested transform compositions | substitution-list forwarding optimization |
| AD dual jaxprs | `ad.py:linearize_jaxpr` | O(E) primal + O(E) tangent | large programs under `grad` | `@weakref_lru_cache` on linearization result |
| Cache key serialization | `cache_key.py:get` | O(S) full IR serialization | large MLIR modules | conditional canonicalization via config flag |
| CompileOptions deepcopy | `cache_key.py:281` | O(C) per cache lookup | frequent cache miss paths | none (known cost accepted) |
| Compilation cache entries | `compilation_cache.py` | O(M * X_avg) where M = max_size | many distinct compilations | LRU eviction with configurable max_size |
| Executable deserialization | `compilation_cache.py:290-294` | O(X) per cache hit | large compiled executables | zstandard/zlib compression |

### 20.3 Legacy Hotspot Families

**Family H1: Trace-to-IR construction path**
- Location: `core.py:eval_jaxpr`, `partial_eval.py:trace_to_jaxpr`, `partial_eval.py:process_call`
- Pattern: sequential equation processing with environment management
- Dominant cost: linear in program size, but residual forwarding in nested calls can compound to O(n^2) through substitution-list chaining
- Memory peak: all tracers stored until jaxpr conversion

**Family H2: Cache key computation path**
- Location: `cache_key.py:get`, `cache_key.py:_hash_serialized_compile_options`, `cache_key.py:_canonicalize_ir`
- Pattern: full IR serialization + deep copy of compile options + tree walk for callback removal
- Dominant cost: O(S) where S = serialized IR module size; deepcopy adds constant-factor overhead
- Memory peak: cloned MLIR module + deepcopied CompileOptions held simultaneously

**Family H3: Transform composition path**
- Location: `ad.py:linearize_jaxpr`, `batching.py:process_primitive`, `partial_eval.py:process_map`
- Pattern: each transform layer wraps/rewrites the program representation
- Dominant cost: multiplicative with transform stack depth (grad adds backward pass, vmap adds per-slice iteration)
- Memory peak: dual jaxpr (primal + tangent) under grad, plus residual lists

**Family H4: DCE and post-trace optimization**
- Location: `partial_eval.py:dce_jaxpr`
- Pattern: reverse pass through equations checking output liveness
- Dominant cost: O(E * K) reverse traversal
- Memory peak: filtered equation lists and used-variable sets

**Family H5: Compilation and executable cache I/O**
- Location: `compilation_cache.py:get_executable_and_time`, `compilation_cache.py:put_executable_and_time`
- Pattern: serialization/deserialization + compression/decompression of compiled executables
- Dominant cost: O(X) where X = executable size; compression adds constant factor
- Memory peak: compressed + decompressed forms held during transition

### 20.4 Complexity Scaling Laws (Legacy)

Transform composition multiplies base evaluation cost:
- `jit` alone: 1x base trace + compilation
- `grad(f)`: 2x base (forward pass + backward pass), plus residual memory
- `vmap(f)`: L x base where L = leading dimension
- `vmap(grad(f))`: L x 2x base (each vmap slice runs full grad)
- Nested `jit(grad(f))`: cache miss = trace + grad + compile; cache hit = O(1) dispatch

### 20.5 Performance Budget Anchors (From Section 8)

Provisional budgets from section 8 (pending empirical calibration):
- transform composition overhead regression: <= +10%
- cache hit path p95 regression: <= +8%
- p99 regression: <= +10%
- peak RSS regression: <= +10%

These budgets apply to the Rust implementation relative to equivalent legacy workload classes and must be validated against hotspot families H1-H5 above.
