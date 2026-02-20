#![forbid(unsafe_code)]

pub mod errors;
pub mod transforms;

pub use errors::ApiError;
pub use transforms::{GradWrapped, JitWrapped, ValueAndGradWrapped, VmapWrapped};
pub use transforms::{grad, jit, value_and_grad, vmap};

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, Value, build_program};

    #[test]
    fn jit_add_scalar() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = jit(jaxpr)
            .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
            .expect("jit should succeed");
        assert_eq!(result, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn grad_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(3.0)])
            .expect("grad should succeed");
        let derivative = result[0]
            .as_f64_scalar()
            .expect("grad output should be scalar");
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn vmap_add_one() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let result = vmap(jaxpr)
            .call(vec![
                Value::vector_i64(&[10, 20, 30]).expect("vector should build"),
            ])
            .expect("vmap should succeed");
        let output = result[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        let values: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64 element"))
            .collect();
        assert_eq!(values, vec![11, 21, 31]);
    }

    #[test]
    fn value_and_grad_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let (value, gradient) = value_and_grad(jaxpr)
            .call(vec![Value::scalar_f64(4.0)])
            .expect("value_and_grad should succeed");

        let val = value[0]
            .as_f64_scalar()
            .expect("value should be scalar");
        assert!((val - 16.0).abs() < 1e-6);

        let grad_val = gradient[0]
            .as_f64_scalar()
            .expect("gradient should be scalar");
        assert!((grad_val - 8.0).abs() < 1e-3);
    }

    #[test]
    fn grad_non_scalar_input_fails() {
        let jaxpr = build_program(ProgramSpec::Square);
        let err = grad(jaxpr)
            .call(vec![
                Value::vector_f64(&[1.0, 2.0]).expect("vector should build"),
            ])
            .expect_err("grad with vector input should fail");
        assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    }

    #[test]
    fn with_mode_hardened() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = jit(jaxpr)
            .with_mode(fj_core::CompatibilityMode::Hardened)
            .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
            .expect("hardened jit should succeed");
        assert_eq!(result, vec![Value::scalar_i64(3)]);
    }
}
