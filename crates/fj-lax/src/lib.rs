#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value, ValueError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    ArityMismatch {
        primitive: Primitive,
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        primitive: Primitive,
        detail: &'static str,
    },
    ShapeMismatch {
        primitive: Primitive,
        left: Shape,
        right: Shape,
    },
    Unsupported {
        primitive: Primitive,
        detail: String,
    },
    InvalidTensor(ValueError),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityMismatch {
                primitive,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "arity mismatch for {}: expected {}, got {}",
                    primitive.as_str(),
                    expected,
                    actual
                )
            }
            Self::TypeMismatch { primitive, detail } => {
                write!(f, "type mismatch for {}: {}", primitive.as_str(), detail)
            }
            Self::ShapeMismatch {
                primitive,
                left,
                right,
            } => {
                write!(
                    f,
                    "shape mismatch for {}: left={:?} right={:?}",
                    primitive.as_str(),
                    left.dims,
                    right.dims
                )
            }
            Self::Unsupported { primitive, detail } => {
                write!(f, "unsupported {} behavior: {}", primitive.as_str(), detail)
            }
            Self::InvalidTensor(err) => write!(f, "invalid tensor: {err}"),
        }
    }
}

impl std::error::Error for EvalError {}

impl From<ValueError> for EvalError {
    fn from(value: ValueError) -> Self {
        Self::InvalidTensor(value)
    }
}

#[inline]
pub fn eval_primitive(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    match primitive {
        Primitive::Add => eval_binary_elementwise(primitive, inputs, |a, b| a + b, |a, b| a + b),
        Primitive::Mul => eval_binary_elementwise(primitive, inputs, |a, b| a * b, |a, b| a * b),
        Primitive::Dot => eval_dot(inputs),
        Primitive::Sin => eval_unary_elementwise(primitive, inputs, f64::sin),
        Primitive::Cos => eval_unary_elementwise(primitive, inputs, f64::cos),
        Primitive::ReduceSum => eval_reduce_sum(inputs),
    }
}

#[inline]
fn eval_binary_elementwise(
    primitive: Primitive,
    inputs: &[Value],
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => Ok(Value::Scalar(binary_literal_op(
            *lhs, *rhs, primitive, &int_op, &float_op,
        )?)),
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.shape != rhs.shape {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }

            let elements = lhs
                .elements
                .iter()
                .copied()
                .zip(rhs.elements.iter().copied())
                .map(|(left, right)| binary_literal_op(left, right, primitive, &int_op, &float_op))
                .collect::<Result<Vec<_>, _>>()?;

            let dtype = infer_dtype(&elements);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                lhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            let elements = rhs
                .elements
                .iter()
                .copied()
                .map(|right| binary_literal_op(*lhs, right, primitive, &int_op, &float_op))
                .collect::<Result<Vec<_>, _>>()?;

            let dtype = infer_dtype(&elements);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            let elements = lhs
                .elements
                .iter()
                .copied()
                .map(|left| binary_literal_op(left, *rhs, primitive, &int_op, &float_op))
                .collect::<Result<Vec<_>, _>>()?;

            let dtype = infer_dtype(&elements);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

#[inline]
fn eval_unary_elementwise(
    primitive: Primitive,
    inputs: &[Value],
    op: impl Fn(f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric scalar",
            })?;
            Ok(Value::scalar_f64(op(value)))
        }
        Value::Tensor(tensor) => {
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(|literal| {
                    literal.as_f64().map(&op).map(Literal::from_f64).ok_or(
                        EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor elements",
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(Value::Tensor(TensorValue::new(
                DType::F64,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

fn eval_dot(inputs: &[Value]) -> Result<Value, EvalError> {
    let primitive = Primitive::Dot;
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => Ok(Value::Scalar(binary_literal_op(
            *lhs,
            *rhs,
            primitive,
            &|a, b| a * b,
            &|a, b| a * b,
        )?)),
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.rank() != 1 || rhs.rank() != 1 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "dot currently supports only rank-1 tensors".to_owned(),
                });
            }
            if lhs.shape != rhs.shape {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }

            if lhs.elements.iter().all(|literal| literal.is_integral())
                && rhs.elements.iter().all(|literal| literal.is_integral())
            {
                let mut sum = 0_i64;
                for (left, right) in lhs.elements.iter().zip(rhs.elements.iter()) {
                    let left_i = left.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "integral dot expected i64 elements",
                    })?;
                    let right_i = right.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "integral dot expected i64 elements",
                    })?;
                    sum += left_i * right_i;
                }
                return Ok(Value::scalar_i64(sum));
            }

            let mut sum = 0.0_f64;
            for (left, right) in lhs.elements.iter().zip(rhs.elements.iter()) {
                let left_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric lhs tensor",
                })?;
                let right_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric rhs tensor",
                })?;
                sum += left_f * right_f;
            }
            Ok(Value::scalar_f64(sum))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "dot expects either two scalars or two vectors".to_owned(),
        }),
    }
}

fn eval_reduce_sum(inputs: &[Value]) -> Result<Value, EvalError> {
    let primitive = Primitive::ReduceSum;
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
                let sum = tensor
                    .elements
                    .iter()
                    .copied()
                    .map(|literal| {
                        literal.as_i64().ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected i64 tensor",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .sum::<i64>();
                return Ok(Value::scalar_i64(sum));
            }

            let sum = tensor
                .elements
                .iter()
                .copied()
                .map(|literal| {
                    literal.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric tensor",
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<f64>();
            Ok(Value::scalar_f64(sum))
        }
    }
}

#[inline]
fn binary_literal_op(
    lhs: Literal,
    rhs: Literal,
    primitive: Primitive,
    int_op: &impl Fn(i64, i64) -> i64,
    float_op: &impl Fn(f64, f64) -> f64,
) -> Result<Literal, EvalError> {
    match (lhs, rhs) {
        (Literal::I64(left), Literal::I64(right)) => Ok(Literal::I64(int_op(left, right))),
        (left, right) => {
            let lhs_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric lhs",
            })?;
            let rhs_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric rhs",
            })?;
            Ok(Literal::from_f64(float_op(lhs_f, rhs_f)))
        }
    }
}

#[inline]
fn infer_dtype(elements: &[Literal]) -> DType {
    if elements
        .iter()
        .all(|literal| matches!(literal, Literal::I64(_)))
    {
        DType::I64
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::Bool(_)))
    {
        DType::Bool
    } else {
        DType::F64
    }
}

#[cfg(test)]
mod tests {
    use super::{EvalError, eval_primitive};
    use fj_core::{Primitive, Value};

    #[test]
    fn add_i64_scalars() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_i64(5)],
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn add_vector_and_scalar_broadcasts() {
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let out = eval_primitive(Primitive::Add, &[input, Value::scalar_i64(2)])
            .expect("broadcasted add should succeed");

        let expected = Value::vector_i64(&[3, 4, 5]).expect("vector value should build");
        assert_eq!(out, expected);
    }

    #[test]
    fn dot_vector_works() {
        let lhs = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let rhs = Value::vector_i64(&[4, 5, 6]).expect("vector value should build");
        let out = eval_primitive(Primitive::Dot, &[lhs, rhs]).expect("dot should succeed");
        assert_eq!(out, Value::scalar_i64(32));
    }

    #[test]
    fn reduce_sum_requires_single_argument() {
        let err = eval_primitive(Primitive::ReduceSum, &[]).expect_err("should fail");
        assert_eq!(
            err,
            EvalError::ArityMismatch {
                primitive: Primitive::ReduceSum,
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn test_lax_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("lax", "add")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_lax_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }
}
