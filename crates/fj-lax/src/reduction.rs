#![forbid(unsafe_code)]

use fj_core::{Primitive, Value};

use crate::EvalError;

/// Generic reduction: reduces all elements of a tensor to a scalar.
pub(crate) fn eval_reduce(
    primitive: Primitive,
    inputs: &[Value],
    int_init: i64,
    float_init: f64,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(*literal)),
        Value::Tensor(tensor) => {
            if tensor.elements.iter().all(|literal| literal.is_integral()) {
                let mut acc = int_init;
                for literal in &tensor.elements {
                    let val = literal.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected i64 tensor",
                    })?;
                    acc = int_op(acc, val);
                }
                return Ok(Value::scalar_i64(acc));
            }

            let mut acc = float_init;
            for literal in &tensor.elements {
                let val = literal.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric tensor",
                })?;
                acc = float_op(acc, val);
            }
            Ok(Value::scalar_f64(acc))
        }
    }
}
