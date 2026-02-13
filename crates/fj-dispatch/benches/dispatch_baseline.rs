use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{
    CompatibilityMode, ProgramSpec, TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
use std::collections::BTreeMap;

fn add_ledger() -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(ProgramSpec::Add2));
    ledger.push_transform(Transform::Jit, "jit-baseline");
    ledger
}

fn benchmark_dispatch(c: &mut Criterion) {
    c.bench_function("dispatch/simple_add", |b| {
        b.iter(|| {
            let response = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: add_ledger(),
                args: vec![Value::scalar_i64(2), Value::scalar_i64(5)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .expect("dispatch benchmark request should succeed");

            assert_eq!(response.outputs, vec![Value::scalar_i64(7)]);
        });
    });
}

criterion_group!(dispatch_benches, benchmark_dispatch);
criterion_main!(dispatch_benches);
