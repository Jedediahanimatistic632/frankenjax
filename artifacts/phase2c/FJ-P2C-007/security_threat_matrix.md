# FJ-P2C-007 Security & Compatibility Threat Matrix: FFI Call Interface

## Packet Scope

FFI call interface — the `unsafe` boundary where FrankenJAX invokes external C functions. This is the **highest security-sensitivity packet** in the system: all other crates use `#![forbid(unsafe_code)]`, but FFI requires controlled `unsafe` for extern function invocation and raw pointer passing.

---

## Threat Matrix

| # | Threat | Severity | Likelihood | Residual Risk | Mitigation | Anchor Ref |
|---|--------|----------|------------|---------------|------------|------------|
| T1 | **Buffer overflow**: FFI callee writes beyond declared output buffer size | Critical | Medium | Low | Output buffers pre-allocated to exact declared size. Guard page or canary byte after buffer (future). Post-call size assertion. | P2C007-A15, P2C007-A02 |
| T2 | **Use-after-free**: Rust drops buffer while FFI callee still references it | Critical | Low | Low | Buffers are stack-local borrows within invoke() unsafe block. Rust borrow checker prevents premature drop on safe side. FFI callee cannot extend borrow beyond call return. | P2C007-A16, P2C007-A04 |
| T3 | **Type confusion**: declared dtype does not match actual memory layout in external function | High | Medium | Medium | Explicit dtype-to-C-type mapping validated at registration time (DType::F64 → c_double, DType::I64 → c_longlong). Mismatch between declared and actual types in external code is **undetectable** from Rust side — documented as caller responsibility. | P2C007-A11, P2C007-A09 |
| T4 | **Code injection**: malicious shared library registered as FFI target | Critical | Low | Low (V1) | V1: no dynamic library loading. All FFI targets are statically linked extern "C" fn pointers compiled into the binary. Future: dlopen targets must be validated against allowlist. | P2C007-A03 |
| T5 | **Double-free**: both Rust and FFI callee attempt to free the same buffer | Critical | Low | Negligible | Ownership contract: FrankenJAX owns all buffers. FFI receives *const/*mut pointers with no ownership transfer. Contract documented in SAFETY comment. | P2C007-A16, P2C007-A05 |
| T6 | **Null pointer dereference**: FFI function pointer is null at call time | High | Low | Negligible | Null check at registration time (FfiRegistry.register). Registered pointer is non-null invariant. Null registration returns FfiError::NullPointer. | P2C007-A03, P2C007-A14 |
| T7 | **Segfault in external code**: FFI function crashes (SIGSEGV, SIGBUS) | Critical | Medium | **Accepted** | Unrecoverable from Rust. Process aborts. No mitigation possible without signal handlers (out of V1 scope). Documented as known risk. Caller must audit external code. | P2C007-A14 |
| T8 | **Thread-safety violation**: FFI function is not thread-safe but called from multiple threads | High | Medium | Medium | V1: FfiRegistry is thread-safe (RwLock). Individual FFI functions are assumed thread-safe by contract. Future: optional `thread_safe: bool` flag per registration. | P2C007-A03 |
| T9 | **Data race on shared buffers**: concurrent FFI calls share input buffer memory | High | Low | Low | Each FfiCall invocation receives independent buffer copies or borrows scoped to a single call. No shared mutable state across concurrent calls. | P2C007-A15, P2C007-A16 |
| T10 | **Integer overflow in buffer size calculation**: product(shape) * dtype_size overflows usize | Medium | Low | Negligible | checked_mul arithmetic for buffer size computation. Overflow returns FfiError::BufferMismatch. | P2C007-A15 |

---

## Compatibility Envelope

| Row | Feature | JAX Behavior | FrankenJAX V1 | Divergence | Risk |
|-----|---------|-------------|---------------|------------|------|
| CE1 | FFI call dispatch | custom_call HLO op via XLA | Direct invocation during eval_jaxpr | Semantic equivalent, different mechanism | None |
| CE2 | FFI target registration | PyCapsule + global registry | FfiRegistry + typed fn pointers | No PyCapsule (no Python) | None |
| CE3 | Buffer protocol | (pointer, shape, dtype) triples | FfiBuffer { data, shape, dtype } struct | Isomorphic representation | None |
| CE4 | dtype mapping | numpy dtype → C types | DType → C ABI types (F64→c_double, I64→c_longlong) | Subset of JAX mappings | Low: fewer types supported |
| CE5 | Error propagation | XlaRuntimeError from C return codes | FfiError from return code protocol | Same protocol, different error type | None |
| CE6 | Memory ownership | JAX owns, C borrows | FrankenJAX owns, C borrows | Identical ownership model | None |
| CE7 | Pure callbacks | Python callable, no side effects | Rust closure, no side effects | Same semantics, different host language | None |
| CE8 | IO callbacks | ordered_effects token chain | EffectContext token threading | Same ordering guarantee | None |
| CE9 | vmap over FFI | Per-element fallback or vectorized=True | Not supported in V1 (error) | Strict subset — JAX programs using vmap+FFI will error | Medium |
| CE10 | grad over FFI | custom_vjp/custom_jvp registration | Not supported in V1 (error) | Strict subset — JAX programs using grad+FFI will error | Medium |
| CE11 | FFI abstract eval | Output shapes/dtypes declared at call site | Output shapes/dtypes in FfiCall equation | Same approach | None |
| CE12 | Dynamic library loading | dlopen/PyCapsule | Not supported in V1 (static linking only) | Strict subset | Low |
| CE13 | FFI timeout | No built-in timeout in JAX | Future V2 feature | V1 matches JAX (no timeout) | None |
| CE14 | Complex type support | complex64/complex128 | Not supported in V1 | Strict subset | Low |
| CE15 | Struct passing | Manual packing | Not supported in V1 | Strict subset | Low |
| CE16 | Callback batching rule | Custom rules per callback | Not supported in V1 | Strict subset | Low |
| CE17 | api_version negotiation | api_version int in custom_call | Not applicable (no HLO) | N/A | None |
| CE18 | backend_config bytes | Opaque bytes to handler | Not supported in V1 | Strict subset | Low |

---

## Fail-Closed Rules

| # | Rule | Trigger | Action |
|---|------|---------|--------|
| FC1 | Unregistered FFI target | call(name) where name not in registry | Return FfiError::TargetNotFound |
| FC2 | Null function pointer | register(name, null_ptr) | Return FfiError::NullPointer, do not add to registry |
| FC3 | Buffer size mismatch | actual_bytes != product(shape) * dtype_size | Return FfiError::BufferMismatch, do not call FFI |
| FC4 | Duplicate registration | register(name, _) where name already registered | Return FfiError::DuplicateTarget |
| FC5 | Non-zero return code | FFI function returns non-zero i32 | Return FfiError::CallFailed with code and message |
| FC6 | Buffer size overflow | product(shape) * dtype_size overflows usize | Return FfiError::BufferMismatch (overflow variant) |
| FC7 | grad over FFI equation | Gradient transform encounters FfiCall | Return TransformExecutionError |
| FC8 | vmap over FFI equation | Batching transform encounters FfiCall | Return TransformExecutionError |
| FC9 | Unsupported dtype at boundary | FfiCall equation uses dtype not in {F64, I64} | Return FfiError::UnsupportedDtype |
| FC10 | Zero-size buffer | FfiCall with empty shape or zero-element tensor | Allow (valid edge case) — pass null pointer with 0 length |

---

## Unsafe Code Audit Checklist

Every `unsafe` block in the FFI module must satisfy ALL of the following:

1. **SAFETY comment** immediately above the unsafe block listing all preconditions
2. **Non-null pointer**: fn_ptr has been validated non-null at registration time
3. **Valid buffer lengths**: all buffer sizes have been validated via checked arithmetic
4. **Ownership contract**: inputs are *const (read borrow), outputs are *mut (write borrow)
5. **Lifetime scope**: all raw pointers are derived from references that outlive the unsafe block
6. **No aliasing**: input and output buffers do not overlap (distinct allocations)
7. **Return code check**: non-zero return is mapped to FfiError before any output buffer is read
8. **Post-call assertion**: output buffer lengths unchanged after call (debug-mode canary)

---

## MIRI Testing Strategy

- **In scope**: FfiBuffer construction, size validation, dtype mapping, registry operations
- **Out of scope**: Actual extern "C" fn calls (MIRI cannot execute foreign functions)
- **Workaround**: Mock FFI functions as Rust closures behind cfg(miri) for memory safety validation
