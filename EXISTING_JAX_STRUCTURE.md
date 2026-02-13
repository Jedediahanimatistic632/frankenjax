# EXISTING_JAX_STRUCTURE

## 1. Legacy Oracle

- Root: `/dp/frankenjax/legacy_jax_code/jax`
- Upstream: `jax-ml/jax`

## 2. High-Value File/Function Anchors

### Transform API entry points

- `jax/_src/api.py`
  - `jit`
  - `vmap`
  - `grad`
  - `value_and_grad`
  - `_vjp`

### JIT and staging/lowering path

- `jax/_src/pjit.py`
  - `JitWrapped`
  - `_parse_jit_arguments`
  - `_run_python_pjit`
- `jax/_src/stages.py`
  - `Traced`

### AD (reverse/forward transform semantics)

- `jax/_src/interpreters/ad.py`
  - `linearize_subtrace`
  - `linearize_jaxpr`
  - `backward_pass3`

### Batching / vmap semantics

- `jax/_src/interpreters/batching.py`
  - `BatchTrace`
  - `BatchTracer`
  - `batch_subtrace`
  - `batch_jaxpr`

### Cache-key and compilation cache semantics

- `jax/_src/cache_key.py`
  - `get`
  - `_hash_serialized_compile_options`
  - `_hash_xla_flags`
  - `_hash_accelerator_config`
- `jax/_src/compiler.py`
  - `compile_or_get_cached`
  - `_resolve_compilation_strategy`
  - `_cache_read`
  - `_cache_write`
- `jax/_src/compilation_cache.py`
  - `get_cache_key`
  - `get_executable_and_time`
  - `put_executable_and_time`

### Dispatch-level memoization and trace caching

- `jax/_src/dispatch.py`
  - `xla_primitive_callable`
  - `_is_supported_cross_host_transfer`
- `jax/_src/interpreters/partial_eval.py`
  - `trace_to_jaxpr`

## 3. Semantic Hotspots (Non-Negotiable)

1. Transform composition semantics for `jit`, `grad`, `vmap`.
2. Deterministic trace/Jaxpr construction and lowering metadata.
3. Cache-key soundness and compatibility-sensitive inputs.
4. Dispatch cache behavior and memoization invariants.
5. Error-path behavior when metadata or shape contracts are incompatible.

## 4. Conformance Fixture Family Anchors

- `jit`: `tests/jax_jit_test.py`
- `grad`: `tests/lax_autodiff_test.py` and `tests/lax_test.py`
- `vmap`: `tests/lax_vmap_test.py` and `tests/lax_vmap_op_test.py`
- `lax primitives`: `tests/lax_test.py`
- `random`: `tests/random_test.py` and `tests/random_lax_test.py`

## 5. Compatibility-Critical Inputs for Cache Keying

The legacy cache key uses deterministic hashing over a canonicalized input bundle that includes:

- canonicalized module bytecode
- backend/version fields
- normalized compile options
- normalized XLA flags
- accelerator topology/device descriptors
- optional runtime custom hook bits

FrankenJAX MUST preserve input-surface semantics for scoped compatibility.

## 6. Security and Reliability Risk Areas

- cache confusion via incompatible/unknown metadata
- transform-order drift leading to semantic mismatch
- malformed shape/graph signatures
- stale cache artifacts or corruption in persistent storage

## 7. Extraction Boundary (Current)

Included now:

- transform API semantics and composition invariants
- deterministic key derivation contract
- core conformance fixture families

Deferred but tracked:

- broad plugin/distributed backend breadth
- long-tail API surfaces outside scoped parity matrix
