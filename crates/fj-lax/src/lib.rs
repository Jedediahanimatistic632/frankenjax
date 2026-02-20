#![forbid(unsafe_code)]

mod arithmetic;
mod comparison;
mod reduction;
mod tensor_ops;
mod type_promotion;

use fj_core::{Primitive, Shape, Value, ValueError};
use std::collections::BTreeMap;

use arithmetic::{eval_binary_elementwise, eval_dot, eval_unary_elementwise, eval_unary_int_or_float};
use comparison::eval_comparison;
use reduction::eval_reduce;
use tensor_ops::{eval_broadcast_in_dim, eval_concatenate, eval_reshape, eval_slice, eval_transpose};

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
pub fn eval_primitive(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    match primitive {
        // Binary arithmetic
        Primitive::Add => eval_binary_elementwise(primitive, inputs, |a, b| a + b, |a, b| a + b),
        Primitive::Sub => eval_binary_elementwise(primitive, inputs, |a, b| a - b, |a, b| a - b),
        Primitive::Mul => eval_binary_elementwise(primitive, inputs, |a, b| a * b, |a, b| a * b),
        Primitive::Max => eval_binary_elementwise(primitive, inputs, |a, b| a.max(b), f64::max),
        Primitive::Min => eval_binary_elementwise(primitive, inputs, |a, b| a.min(b), f64::min),
        Primitive::Pow => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| (a as f64).powf(b as f64) as i64,
            f64::powf,
        ),
        // Unary arithmetic
        Primitive::Neg => eval_unary_int_or_float(primitive, inputs, |x| -x, |x| -x),
        Primitive::Abs => eval_unary_int_or_float(primitive, inputs, i64::abs, f64::abs),
        Primitive::Exp => eval_unary_elementwise(primitive, inputs, f64::exp),
        Primitive::Log => eval_unary_elementwise(primitive, inputs, f64::ln),
        Primitive::Sqrt => eval_unary_elementwise(primitive, inputs, f64::sqrt),
        Primitive::Rsqrt => eval_unary_elementwise(primitive, inputs, |x| 1.0 / x.sqrt()),
        Primitive::Floor => eval_unary_elementwise(primitive, inputs, f64::floor),
        Primitive::Ceil => eval_unary_elementwise(primitive, inputs, f64::ceil),
        Primitive::Round => eval_unary_elementwise(primitive, inputs, f64::round),
        // Trigonometric
        Primitive::Sin => eval_unary_elementwise(primitive, inputs, f64::sin),
        Primitive::Cos => eval_unary_elementwise(primitive, inputs, f64::cos),
        // Dot product
        Primitive::Dot => eval_dot(inputs),
        // Comparison
        Primitive::Eq => eval_comparison(primitive, inputs, |a, b| a == b, |a, b| a == b),
        Primitive::Ne => eval_comparison(primitive, inputs, |a, b| a != b, |a, b| a != b),
        Primitive::Lt => eval_comparison(primitive, inputs, |a, b| a < b, |a, b| a < b),
        Primitive::Le => eval_comparison(primitive, inputs, |a, b| a <= b, |a, b| a <= b),
        Primitive::Gt => eval_comparison(primitive, inputs, |a, b| a > b, |a, b| a > b),
        Primitive::Ge => eval_comparison(primitive, inputs, |a, b| a >= b, |a, b| a >= b),
        // Reductions
        Primitive::ReduceSum => {
            eval_reduce(primitive, inputs, 0_i64, 0.0, |a, b| a + b, |a, b| a + b)
        }
        Primitive::ReduceMax => eval_reduce(
            primitive,
            inputs,
            i64::MIN,
            f64::NEG_INFINITY,
            i64::max,
            f64::max,
        ),
        Primitive::ReduceMin => eval_reduce(
            primitive,
            inputs,
            i64::MAX,
            f64::INFINITY,
            i64::min,
            f64::min,
        ),
        Primitive::ReduceProd => {
            eval_reduce(primitive, inputs, 1_i64, 1.0, |a, b| a * b, |a, b| a * b)
        }
        // Shape manipulation
        Primitive::Reshape => eval_reshape(inputs, params),
        Primitive::Transpose => eval_transpose(inputs, params),
        Primitive::BroadcastInDim => eval_broadcast_in_dim(inputs, params),
        Primitive::Concatenate => eval_concatenate(inputs, params),
        Primitive::Slice => eval_slice(inputs, params),
        // Not yet implemented
        Primitive::Gather | Primitive::Scatter => Err(EvalError::Unsupported {
            primitive,
            detail: "runtime kernel not implemented yet for this primitive".to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{EvalError, eval_primitive};
    use fj_core::{DType, Primitive, Value};
    use std::collections::BTreeMap;

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    #[test]
    fn add_i64_scalars() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_i64(5)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn add_vector_and_scalar_broadcasts() {
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let out = eval_primitive(Primitive::Add, &[input, Value::scalar_i64(2)], &no_params())
            .expect("broadcasted add should succeed");

        let expected = Value::vector_i64(&[3, 4, 5]).expect("vector value should build");
        assert_eq!(out, expected);
    }

    #[test]
    fn sub_i64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn sub_f64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_f64(5.5), Value::scalar_f64(2.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.5).abs() < 1e-10);
    }

    #[test]
    fn neg_i64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_i64(7)], &no_params());
        assert_eq!(out, Ok(Value::scalar_i64(-7)));
    }

    #[test]
    fn neg_f64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_f64(3.5)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - (-3.5)).abs() < 1e-10);
    }

    #[test]
    fn abs_negative_i64() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_i64(-42)], &no_params());
        assert_eq!(out, Ok(Value::scalar_i64(42)));
    }

    #[test]
    fn abs_negative_f64() {
        let out =
            eval_primitive(Primitive::Abs, &[Value::scalar_f64(-2.78)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 2.78).abs() < 1e-10);
    }

    #[test]
    fn max_i64_scalars() {
        let out = eval_primitive(
            Primitive::Max,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn min_i64_scalars() {
        let out = eval_primitive(
            Primitive::Min,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(3)));
    }

    #[test]
    fn exp_scalar() {
        let out = eval_primitive(Primitive::Exp, &[Value::scalar_f64(1.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_scalar() {
        let out = eval_primitive(
            Primitive::Log,
            &[Value::scalar_f64(std::f64::consts::E)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sqrt_scalar() {
        let out = eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(9.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn rsqrt_scalar() {
        let out =
            eval_primitive(Primitive::Rsqrt, &[Value::scalar_f64(4.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn floor_scalar() {
        let out =
            eval_primitive(Primitive::Floor, &[Value::scalar_f64(3.7)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn ceil_scalar() {
        let out = eval_primitive(Primitive::Ceil, &[Value::scalar_f64(3.2)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn round_scalar() {
        let out =
            eval_primitive(Primitive::Round, &[Value::scalar_f64(3.5)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn pow_f64_scalars() {
        let out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 8.0).abs() < 1e-10);
    }

    #[test]
    fn eq_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn ne_i64_scalars() {
        let out = eval_primitive(
            Primitive::Ne,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn lt_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_i64(3), Value::scalar_i64(5)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_i64(5), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn le_ge_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Le,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Ge,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn gt_f64_scalars() {
        let out = eval_primitive(
            Primitive::Gt,
            &[Value::scalar_f64(3.5), Value::scalar_f64(2.0)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn comparison_on_vectors() {
        let lhs = Value::vector_i64(&[1, 2, 3]).unwrap();
        let rhs = Value::vector_i64(&[2, 2, 1]).unwrap();
        let out = eval_primitive(Primitive::Lt, &[lhs, rhs], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 3);
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true));
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false));
            assert_eq!(t.elements[2], fj_core::Literal::Bool(false));
        } else {
            panic!("expected tensor output for vector comparison");
        }
    }

    #[test]
    fn reduce_max_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(9));
    }

    #[test]
    fn reduce_min_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(1));
    }

    #[test]
    fn reduce_prod_vector() {
        let input = Value::vector_i64(&[2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(24));
    }

    #[test]
    fn neg_vector() {
        let input = Value::vector_i64(&[1, -2, 3]).unwrap();
        let out = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
        let expected = Value::vector_i64(&[-1, 2, -3]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dot_vector_works() {
        let lhs = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let rhs = Value::vector_i64(&[4, 5, 6]).expect("vector value should build");
        let out =
            eval_primitive(Primitive::Dot, &[lhs, rhs], &no_params()).expect("dot should succeed");
        assert_eq!(out, Value::scalar_i64(32));
    }

    #[test]
    fn reduce_sum_requires_single_argument() {
        let err = eval_primitive(Primitive::ReduceSum, &[], &no_params()).expect_err("should fail");
        assert_eq!(
            err,
            EvalError::ArityMismatch {
                primitive: Primitive::ReduceSum,
                expected: 1,
                actual: 0,
            }
        );
    }

    // --- Shape manipulation tests ---

    #[test]
    fn reshape_vector_to_matrix() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,3".into());
        let out = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            assert_eq!(t.elements.len(), 6);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn reshape_with_inferred_dim() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,-1".into());
        let out = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn transpose_2d() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let input = fj_core::TensorValue::new(
            DType::I64,
            fj_core::Shape { dims: vec![2, 3] },
            vec![
                fj_core::Literal::I64(1),
                fj_core::Literal::I64(2),
                fj_core::Literal::I64(3),
                fj_core::Literal::I64(4),
                fj_core::Literal::I64(5),
                fj_core::Literal::I64(6),
            ],
        )
        .unwrap();
        let out =
            eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 2]);
            assert_eq!(t.elements[0], fj_core::Literal::I64(1));
            assert_eq!(t.elements[1], fj_core::Literal::I64(4));
            assert_eq!(t.elements[2], fj_core::Literal::I64(2));
            assert_eq!(t.elements[3], fj_core::Literal::I64(5));
            assert_eq!(t.elements[4], fj_core::Literal::I64(3));
            assert_eq!(t.elements[5], fj_core::Literal::I64(6));
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn broadcast_in_dim_scalar_to_vector() {
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3".into());
        let out =
            eval_primitive(Primitive::BroadcastInDim, &[Value::scalar_i64(5)], &params).unwrap();
        let expected = Value::vector_i64(&[5, 5, 5]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn broadcast_in_dim_vector_to_matrix() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,3".into());
        params.insert("broadcast_dimensions".into(), "1".into());
        let out = eval_primitive(Primitive::BroadcastInDim, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            // Row 0: [1,2,3], Row 1: [1,2,3]
            assert_eq!(t.elements[0], fj_core::Literal::I64(1));
            assert_eq!(t.elements[1], fj_core::Literal::I64(2));
            assert_eq!(t.elements[2], fj_core::Literal::I64(3));
            assert_eq!(t.elements[3], fj_core::Literal::I64(1));
            assert_eq!(t.elements[4], fj_core::Literal::I64(2));
            assert_eq!(t.elements[5], fj_core::Literal::I64(3));
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn concatenate_vectors() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[3, 4, 5]).unwrap();
        let out = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
        let expected = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn slice_vector() {
        let input = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "4".into());
        let out = eval_primitive(Primitive::Slice, &[input], &params).unwrap();
        let expected = Value::vector_i64(&[20, 30, 40]).unwrap();
        assert_eq!(out, expected);
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
