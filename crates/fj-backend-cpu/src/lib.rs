//! CPU backend implementation for FrankenJAX.
//!
//! Wraps the existing `eval_jaxpr` interpreter as a `Backend` implementation.
//! CPU is the baseline backend — always available, no external dependencies.
//!
//! Legacy anchor: P2C006-A17 (CpuBackend), P2C006-A02 (_discover_backends).

#![forbid(unsafe_code)]

mod executor;

pub use executor::CpuBackend;
