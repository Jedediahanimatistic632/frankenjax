#![forbid(unsafe_code)]

mod arithmetic;
mod comparison;
mod reduction;
mod tensor_ops;
pub mod threefry;
mod type_promotion;

use fj_core::{Literal, Primitive, Shape, TensorValue, Value, ValueError};
use std::collections::BTreeMap;

use arithmetic::{
    erf_approx, eval_binary_elementwise, eval_clamp, eval_dot, eval_select, eval_unary_elementwise,
    eval_unary_int_or_float,
};

use comparison::eval_comparison;
use reduction::{eval_cumulative, eval_reduce_axes};
use tensor_ops::{
    eval_argsort, eval_broadcast_in_dim, eval_concatenate, eval_conv, eval_dynamic_slice,
    eval_dynamic_update_slice, eval_gather, eval_iota, eval_one_hot, eval_pad, eval_reshape,
    eval_scatter, eval_slice, eval_sort, eval_transpose,
};

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
    MaxIterationsExceeded {
        primitive: Primitive,
        max_iterations: usize,
    },
    ShapeChanged {
        primitive: Primitive,
        detail: String,
    },
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
            Self::MaxIterationsExceeded {
                primitive,
                max_iterations,
            } => {
                write!(
                    f,
                    "{} exceeded max iterations ({})",
                    primitive.as_str(),
                    max_iterations
                )
            }
            Self::ShapeChanged { primitive, detail } => {
                write!(
                    f,
                    "{} body changed carry shape: {}",
                    primitive.as_str(),
                    detail
                )
            }
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
        Primitive::Tan => eval_unary_elementwise(primitive, inputs, f64::tan),
        Primitive::Asin => eval_unary_elementwise(primitive, inputs, f64::asin),
        Primitive::Acos => eval_unary_elementwise(primitive, inputs, f64::acos),
        Primitive::Atan => eval_unary_elementwise(primitive, inputs, f64::atan),
        // Hyperbolic
        Primitive::Sinh => eval_unary_elementwise(primitive, inputs, f64::sinh),
        Primitive::Cosh => eval_unary_elementwise(primitive, inputs, f64::cosh),
        Primitive::Tanh => eval_unary_elementwise(primitive, inputs, f64::tanh),
        // Additional math
        Primitive::Expm1 => eval_unary_elementwise(primitive, inputs, f64::exp_m1),
        Primitive::Log1p => eval_unary_elementwise(primitive, inputs, f64::ln_1p),
        Primitive::Sign => eval_unary_int_or_float(
            primitive,
            inputs,
            |x| x.signum(),
            |x| {
                if x.is_nan() {
                    f64::NAN
                } else if x == 0.0 {
                    x
                } else {
                    x.signum()
                }
            },
        ),
        Primitive::Square => eval_unary_int_or_float(primitive, inputs, |x| x * x, |x| x * x),
        Primitive::Reciprocal => eval_unary_elementwise(primitive, inputs, |x| 1.0 / x),
        Primitive::Logistic => {
            eval_unary_elementwise(primitive, inputs, |x| 1.0 / (1.0 + (-x).exp()))
        }
        Primitive::Erf => eval_unary_elementwise(primitive, inputs, erf_approx),
        Primitive::Erfc => eval_unary_elementwise(primitive, inputs, |x| 1.0 - erf_approx(x)),
        // Binary math
        Primitive::Div => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| if b != 0 { a / b } else { 0 },
            |a, b| a / b,
        ),
        Primitive::Rem => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| if b != 0 { a % b } else { 0 },
            |a, b| a % b,
        ),
        Primitive::Atan2 => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| (a as f64).atan2(b as f64) as i64,
            f64::atan2,
        ),
        // Selection
        Primitive::Select => eval_select(primitive, inputs),
        // Dot product
        Primitive::Dot => eval_dot(inputs),
        // Comparison
        Primitive::Eq => eval_comparison(primitive, inputs, |a, b| a == b, |a, b| a == b),
        Primitive::Ne => eval_comparison(primitive, inputs, |a, b| a != b, |a, b| a != b),
        Primitive::Lt => eval_comparison(primitive, inputs, |a, b| a < b, |a, b| a < b),
        Primitive::Le => eval_comparison(primitive, inputs, |a, b| a <= b, |a, b| a <= b),
        Primitive::Gt => eval_comparison(primitive, inputs, |a, b| a > b, |a, b| a > b),
        Primitive::Ge => eval_comparison(primitive, inputs, |a, b| a >= b, |a, b| a >= b),
        // Reductions (axis-aware)
        Primitive::ReduceSum => eval_reduce_axes(
            primitive,
            inputs,
            params,
            0_i64,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        ),
        Primitive::ReduceMax => eval_reduce_axes(
            primitive,
            inputs,
            params,
            i64::MIN,
            f64::NEG_INFINITY,
            i64::max,
            f64::max,
        ),
        Primitive::ReduceMin => eval_reduce_axes(
            primitive,
            inputs,
            params,
            i64::MAX,
            f64::INFINITY,
            i64::min,
            f64::min,
        ),
        Primitive::ReduceProd => eval_reduce_axes(
            primitive,
            inputs,
            params,
            1_i64,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        ),
        // Clamp: clamp(x, lo, hi) = min(max(x, lo), hi)
        Primitive::Clamp => eval_clamp(primitive, inputs),
        // Shape manipulation
        Primitive::Reshape => eval_reshape(inputs, params),
        Primitive::Transpose => eval_transpose(inputs, params),
        Primitive::BroadcastInDim => eval_broadcast_in_dim(inputs, params),
        Primitive::Concatenate => eval_concatenate(inputs, params),
        Primitive::Pad => eval_pad(inputs, params),
        Primitive::Slice => eval_slice(inputs, params),
        Primitive::DynamicSlice => eval_dynamic_slice(inputs, params),
        Primitive::Gather => eval_gather(inputs, params),
        Primitive::Scatter => eval_scatter(inputs, params),
        // Iota: generate index sequence
        Primitive::Iota => eval_iota(inputs, params),
        // One-hot encoding
        Primitive::OneHot => eval_one_hot(inputs, params),
        // Dynamic update slice
        Primitive::DynamicUpdateSlice => eval_dynamic_update_slice(inputs, params),
        // Cumulative operations
        Primitive::Cumsum => eval_cumulative(
            primitive,
            inputs,
            params,
            0_i64,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        ),
        Primitive::Cumprod => eval_cumulative(
            primitive,
            inputs,
            params,
            1_i64,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        ),
        // Sorting
        Primitive::Sort => eval_sort(primitive, inputs, params),
        Primitive::Argsort => eval_argsort(primitive, inputs, params),
        // Convolution
        Primitive::Conv => eval_conv(primitive, inputs, params),
        // Control flow
        Primitive::Cond => eval_cond(primitive, inputs),
        Primitive::Scan => eval_scan(primitive, inputs, params),
        Primitive::While => eval_while_loop(primitive, inputs, params),
        // Bitwise
        Primitive::BitwiseAnd => eval_bitwise_binary(primitive, inputs, |a, b| a & b),
        Primitive::BitwiseOr => eval_bitwise_binary(primitive, inputs, |a, b| a | b),
        Primitive::BitwiseXor => eval_bitwise_binary(primitive, inputs, |a, b| a ^ b),
        Primitive::BitwiseNot => eval_bitwise_unary(primitive, inputs, |a| !a),
        Primitive::ShiftLeft => eval_bitwise_binary(primitive, inputs, |a, b| a << b),
        Primitive::ShiftRight => eval_bitwise_binary(primitive, inputs, |a, b| a >> b),
        // Windowed reduction (pooling)
        Primitive::ReduceWindow => eval_reduce_window(primitive, inputs, params),
        // Integer intrinsics
        Primitive::PopulationCount => {
            eval_bitwise_unary(primitive, inputs, |a| a.count_ones() as i64)
        }
        Primitive::CountLeadingZeros => {
            eval_bitwise_unary(primitive, inputs, |a| a.leading_zeros() as i64)
        }
    }
}

/// Evaluate Cond: select between two operands based on a boolean predicate.
///
/// inputs: [predicate, true_value, false_value]
/// Returns true_value if predicate is true, false_value otherwise.
fn eval_cond(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }
    let pred = match &inputs[0] {
        Value::Scalar(fj_core::Literal::Bool(b)) => *b,
        Value::Scalar(fj_core::Literal::I64(v)) => *v != 0,
        Value::Scalar(fj_core::Literal::F64Bits(bits)) => f64::from_bits(*bits) != 0.0,
        _ => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cond predicate must be a scalar".to_owned(),
            });
        }
    };
    if pred {
        Ok(inputs[1].clone())
    } else {
        Ok(inputs[2].clone())
    }
}

/// Evaluate Scan: iterate a body operation over slices of a tensor, threading carry state.
///
/// inputs: [init_carry, xs_tensor]
///   - init_carry: initial carry value (scalar or tensor)
///   - xs_tensor: tensor whose leading axis is scanned over
///
/// params:
///   - "body_op": the primitive to apply per iteration, e.g. "add", "mul"
///     The body computes: new_carry = body_op(carry, x_slice)
///   - "length": optional explicit scan length (inferred from xs if absent)
///   - "reverse": "true" to scan in reverse order (default: "false")
///
/// Returns the final carry value after all iterations (legacy single-value API).
fn eval_scan(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() < 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let init_carry = &inputs[0];
    let xs = &inputs[1];

    // Determine body operation (defaults to Add if not specified)
    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op = match body_op_name {
        "add" => Primitive::Add,
        "sub" => Primitive::Sub,
        "mul" => Primitive::Mul,
        "max" => Primitive::Max,
        "min" => Primitive::Min,
        other => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("unsupported scan body_op: {other}"),
            });
        }
    };

    let reverse = params.get("reverse").map(|s| s == "true").unwrap_or(false);

    // Extract slices from xs along leading axis
    match xs {
        Value::Scalar(_) => {
            // Single element scan — just apply body_op to carry and xs
            eval_primitive(body_op, &[init_carry.clone(), xs.clone()], &BTreeMap::new())
        }
        Value::Tensor(t) => {
            let leading_dim = t.shape.dims[0] as usize;
            if leading_dim == 0 {
                return Ok(init_carry.clone());
            }

            let mut carry = init_carry.clone();
            let indices: Vec<usize> = if reverse {
                (0..leading_dim).rev().collect()
            } else {
                (0..leading_dim).collect()
            };

            for i in indices {
                let x_slice = t.slice_axis0(i).map_err(EvalError::InvalidTensor)?;
                carry = eval_primitive(body_op, &[carry, x_slice], &BTreeMap::new())?;
            }

            Ok(carry)
        }
    }
}

/// Evaluate scan with a functional body: `body_fn(carry, x_i) -> (new_carry, y_i)`.
///
/// This is the full JAX-compatible scan API:
///   `scan(body_fn, init_carry, xs) -> (final_carry, stacked_ys)`
///
/// The body function receives the current carry and one slice of xs, and
/// returns a new carry and an output value. Outputs are collected and stacked.
///
/// - `init_carry`: initial carry values (one or more)
/// - `xs`: tensor to scan over (leading axis is iterated)
/// - `body_fn`: `(carry_values, x_slice) -> (new_carry_values, output_values)`
/// - `reverse`: if true, iterate xs in reverse order
///
/// Returns `(final_carry_values, stacked_ys)` where `stacked_ys` are stacked
/// along a new leading axis.
pub fn eval_scan_functional<B>(
    init_carry: Vec<Value>,
    xs: &Value,
    mut body_fn: B,
    reverse: bool,
) -> Result<(Vec<Value>, Vec<Value>), EvalError>
where
    B: FnMut(Vec<Value>, Value) -> Result<(Vec<Value>, Vec<Value>), EvalError>,
{
    let slices = scan_extract_slices(xs)?;

    if slices.is_empty() {
        // Empty scan: return init carry, no outputs
        return Ok((init_carry, vec![]));
    }

    let indices: Vec<usize> = if reverse {
        (0..slices.len()).rev().collect()
    } else {
        (0..slices.len()).collect()
    };

    let mut carry = init_carry;
    let mut per_output_values: Vec<Vec<Value>> = Vec::new();

    for i in indices {
        let x_slice = slices[i].clone();
        let (new_carry, ys) = body_fn(carry, x_slice)?;
        carry = new_carry;

        // Initialize per-output collectors on first iteration
        if per_output_values.is_empty() {
            per_output_values = vec![Vec::with_capacity(slices.len()); ys.len()];
        }

        for (out_idx, y) in ys.into_iter().enumerate() {
            if out_idx < per_output_values.len() {
                per_output_values[out_idx].push(y);
            }
        }
    }

    // Stack each output along a new leading axis
    let stacked_ys: Vec<Value> = per_output_values
        .into_iter()
        .map(|values| {
            if values.is_empty() {
                return Ok(Value::scalar_f64(0.0));
            }
            // Check if all values are scalars
            let all_scalar = values.iter().all(|v| matches!(v, Value::Scalar(_)));
            if all_scalar {
                // Stack scalars into a vector
                let elements: Vec<Literal> = values
                    .iter()
                    .map(|v| match v {
                        Value::Scalar(lit) => *lit,
                        _ => unreachable!(),
                    })
                    .collect();
                let dtype = Value::Scalar(elements[0]).dtype();
                let len = elements.len() as u32;
                TensorValue::new(dtype, Shape { dims: vec![len] }, elements)
                    .map(Value::Tensor)
                    .map_err(EvalError::InvalidTensor)
            } else {
                TensorValue::stack_axis0(&values)
                    .map(Value::Tensor)
                    .map_err(EvalError::InvalidTensor)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((carry, stacked_ys))
}

/// Extract per-element slices from the leading axis of a value.
fn scan_extract_slices(xs: &Value) -> Result<Vec<Value>, EvalError> {
    match xs {
        Value::Scalar(_) => Ok(vec![xs.clone()]),
        Value::Tensor(t) => {
            let leading_dim = t.shape.dims[0] as usize;
            let mut slices = Vec::with_capacity(leading_dim);
            for i in 0..leading_dim {
                slices.push(t.slice_axis0(i).map_err(EvalError::InvalidTensor)?);
            }
            Ok(slices)
        }
    }
}

/// Evaluate While: iterate a body operation on carry while a condition holds.
///
/// inputs: [init_carry, step_value, threshold]
///   - init_carry: initial carry value (scalar or tensor)
///   - step_value: value applied via body_op each iteration
///   - threshold: value compared against via cond_op
///
/// params:
///   - "body_op": the primitive to apply per iteration, e.g. "add", "mul", "div", "pow"
///     The body computes: new_carry = body_op(carry, step_value)
///   - "cond_op": comparison primitive, e.g. "lt", "le", "gt", "ge", "ne", "eq"
///     Loop continues while: cond_op(carry, threshold) is true
///   - "max_iter": safety limit on iterations (default: 1000)
///
/// Returns the final carry value when the condition becomes false.
/// Returns `MaxIterationsExceeded` if the limit is reached without the condition
/// becoming false.
fn eval_while_loop(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() < 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    let init_carry = &inputs[0];
    let step_value = &inputs[1];
    let threshold = &inputs[2];

    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op = parse_while_body_op(primitive, body_op_name)?;

    let cond_op_name = params.get("cond_op").map(|s| s.as_str()).unwrap_or("lt");
    let cond_op = parse_while_cond_op(primitive, cond_op_name)?;

    let max_iter: usize = params
        .get("max_iter")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let init_shape = value_shape_fingerprint(init_carry);

    let cond_fn = |carry: &Value| -> Result<bool, EvalError> {
        let cond_result = eval_primitive(
            cond_op,
            &[carry.clone(), threshold.clone()],
            &BTreeMap::new(),
        )?;
        Ok(value_to_bool(&cond_result))
    };

    let body_fn = |carry: Value| -> Result<Value, EvalError> {
        eval_primitive(body_op, &[carry, step_value.clone()], &BTreeMap::new())
    };

    eval_while_loop_core(
        primitive,
        init_carry.clone(),
        &init_shape,
        max_iter,
        cond_fn,
        body_fn,
    )
}

/// Evaluate a while loop with arbitrary condition and body functions.
///
/// This is the core implementation used by both the param-based `eval_while_loop`
/// and can be called directly by higher-level evaluators (e.g., fj-dispatch)
/// that provide sub_jaxpr-based condition and body functions.
pub fn eval_while_loop_functional<C, B>(
    init_carry: Vec<Value>,
    max_iterations: usize,
    mut cond_fn: C,
    mut body_fn: B,
) -> Result<Vec<Value>, EvalError>
where
    C: FnMut(&[Value]) -> Result<bool, EvalError>,
    B: FnMut(Vec<Value>) -> Result<Vec<Value>, EvalError>,
{
    let init_shapes: Vec<String> = init_carry.iter().map(value_shape_fingerprint).collect();
    let mut carry = init_carry;

    for _ in 0..max_iterations {
        if !cond_fn(&carry)? {
            return Ok(carry);
        }
        carry = body_fn(carry)?;
        // Verify carry shape is preserved
        for (i, (new_shape, orig_shape)) in carry
            .iter()
            .map(value_shape_fingerprint)
            .zip(init_shapes.iter())
            .enumerate()
        {
            if new_shape != *orig_shape {
                return Err(EvalError::ShapeChanged {
                    primitive: Primitive::While,
                    detail: format!(
                        "carry element {i} changed shape from {orig_shape} to {new_shape}"
                    ),
                });
            }
        }
    }
    Err(EvalError::MaxIterationsExceeded {
        primitive: Primitive::While,
        max_iterations,
    })
}

/// Internal while_loop core that handles single-value carry.
fn eval_while_loop_core<C, B>(
    primitive: Primitive,
    init: Value,
    init_shape: &str,
    max_iter: usize,
    mut cond_fn: C,
    mut body_fn: B,
) -> Result<Value, EvalError>
where
    C: FnMut(&Value) -> Result<bool, EvalError>,
    B: FnMut(Value) -> Result<Value, EvalError>,
{
    let mut carry = init;

    for _ in 0..max_iter {
        if !cond_fn(&carry)? {
            return Ok(carry);
        }
        carry = body_fn(carry)?;
        // Verify shape preservation
        let new_shape = value_shape_fingerprint(&carry);
        if new_shape != init_shape {
            return Err(EvalError::ShapeChanged {
                primitive,
                detail: format!("carry changed from {init_shape} to {new_shape}"),
            });
        }
    }
    Err(EvalError::MaxIterationsExceeded {
        primitive,
        max_iterations: max_iter,
    })
}

fn parse_while_body_op(primitive: Primitive, name: &str) -> Result<Primitive, EvalError> {
    match name {
        "add" => Ok(Primitive::Add),
        "sub" => Ok(Primitive::Sub),
        "mul" => Ok(Primitive::Mul),
        "div" => Ok(Primitive::Div),
        "pow" => Ok(Primitive::Pow),
        other => Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported while body_op: {other}"),
        }),
    }
}

fn parse_while_cond_op(primitive: Primitive, name: &str) -> Result<Primitive, EvalError> {
    match name {
        "lt" => Ok(Primitive::Lt),
        "le" => Ok(Primitive::Le),
        "gt" => Ok(Primitive::Gt),
        "ge" => Ok(Primitive::Ge),
        "ne" => Ok(Primitive::Ne),
        "eq" => Ok(Primitive::Eq),
        other => Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported while cond_op: {other}"),
        }),
    }
}

/// Extract a boolean-ish value from a comparison result.
fn value_to_bool(v: &Value) -> bool {
    match v {
        Value::Scalar(fj_core::Literal::Bool(b)) => *b,
        Value::Scalar(fj_core::Literal::I64(v)) => *v != 0,
        Value::Scalar(fj_core::Literal::F64Bits(bits)) => f64::from_bits(*bits) != 0.0,
        _ => false,
    }
}

/// Compute a simple shape fingerprint for shape-preservation checks.
fn value_shape_fingerprint(v: &Value) -> String {
    match v {
        Value::Scalar(lit) => {
            let kind = match lit {
                fj_core::Literal::I64(_) => "i64",
                fj_core::Literal::Bool(_) => "bool",
                fj_core::Literal::F64Bits(_) => "f64",
                fj_core::Literal::Complex64Bits(_, _) => "c64",
                fj_core::Literal::Complex128Bits(_, _) => "c128",
            };
            format!("scalar:{kind}")
        }
        Value::Tensor(t) => format!("tensor:{:?}:{:?}", t.dtype, t.shape.dims),
    }
}

/// Evaluate a binary bitwise operation on integer values.
fn eval_bitwise_binary(
    primitive: Primitive,
    inputs: &[Value],
    op: fn(i64, i64) -> i64,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }
    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(fj_core::Literal::I64(a)), Value::Scalar(fj_core::Literal::I64(b))) => {
            Ok(Value::scalar_i64(op(*a, *b)))
        }
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: a.shape.clone(),
                    right: b.shape.clone(),
                });
            }
            let elements: Result<Vec<_>, _> = a
                .elements
                .iter()
                .zip(b.elements.iter())
                .map(|(ea, eb)| match (ea, eb) {
                    (fj_core::Literal::I64(va), fj_core::Literal::I64(vb)) => {
                        Ok(fj_core::Literal::I64(op(*va, *vb)))
                    }
                    _ => Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise ops require integer types",
                    }),
                })
                .collect();
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I64, a.shape.clone(), elements?)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: "bitwise ops require integer types",
        }),
    }
}

/// Evaluate a unary bitwise operation on integer values.
fn eval_bitwise_unary(
    primitive: Primitive,
    inputs: &[Value],
    op: fn(i64) -> i64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }
    match &inputs[0] {
        Value::Scalar(fj_core::Literal::I64(a)) => Ok(Value::scalar_i64(op(*a))),
        Value::Tensor(t) => {
            let elements: Result<Vec<_>, _> = t
                .elements
                .iter()
                .map(|e| match e {
                    fj_core::Literal::I64(v) => Ok(fj_core::Literal::I64(op(*v))),
                    _ => Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise ops require integer types",
                    }),
                })
                .collect();
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I64, t.shape.clone(), elements?)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: "bitwise ops require integer types",
        }),
    }
}

/// Evaluate ReduceWindow: apply a reduction over sliding windows of a tensor.
///
/// inputs: [tensor]
/// params:
///   - "reduce_op": "sum", "max", "min" (default: "sum")
///   - "window_dimensions": comma-separated window sizes per dimension, e.g. "2,2"
///   - "window_strides": comma-separated strides, e.g. "1,1" (default: all 1s)
///   - "padding": "valid" or "same" (default: "valid")
///
/// Returns the reduced tensor with output shape determined by window/stride/padding.
fn eval_reduce_window(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: 0,
        });
    }

    let tensor = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => return Ok(inputs[0].clone()), // scalar passthrough
    };

    let rank = tensor.shape.rank();

    // Parse window dimensions
    let window_dims: Vec<usize> = params
        .get("window_dimensions")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2; rank]);

    if window_dims.len() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "window_dimensions length {} doesn't match tensor rank {}",
                window_dims.len(),
                rank
            ),
        });
    }

    // Parse strides
    let strides: Vec<usize> = params
        .get("window_strides")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1; rank]);

    // Determine reduction operation
    let reduce_op = params.get("reduce_op").map(|s| s.as_str()).unwrap_or("sum");

    let padding = params.get("padding").map(|s| s.as_str()).unwrap_or("valid");

    // Calculate output dimensions
    let mut out_dims: Vec<u32> = Vec::with_capacity(rank);
    for d in 0..rank {
        let input_dim = tensor.shape.dims[d] as usize;
        let win = window_dims[d];
        let stride = strides[d];
        let out_dim = match padding {
            "same" => input_dim.div_ceil(stride),
            _ => {
                // "valid"
                if input_dim >= win {
                    (input_dim - win) / stride + 1
                } else {
                    0
                }
            }
        };
        out_dims.push(out_dim as u32);
    }

    let total_output: usize = out_dims.iter().map(|d| *d as usize).product();
    if total_output == 0 {
        return Ok(Value::Tensor(
            TensorValue::new(
                tensor.dtype,
                Shape {
                    dims: out_dims.clone(),
                },
                vec![],
            )
            .map_err(EvalError::InvalidTensor)?,
        ));
    }

    // For 1D case: straightforward sliding window
    // For N-D: use multi-dimensional indexing
    let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();

    let mut output_elements = Vec::with_capacity(total_output);

    // Iterate over all output positions using multi-dimensional index
    let out_dims_usize: Vec<usize> = out_dims.iter().map(|d| *d as usize).collect();
    let mut out_idx = vec![0usize; rank];

    for _ in 0..total_output {
        // For this output position, compute the window
        let init_val = match reduce_op {
            "max" => f64::NEG_INFINITY,
            "min" => f64::INFINITY,
            _ => 0.0, // sum
        };
        let mut accum = init_val;

        // Iterate over all positions within the window
        let mut win_idx = vec![0usize; rank];
        let win_total: usize = window_dims.iter().product();

        for _ in 0..win_total {
            // Compute input index for this window position
            let mut in_bounds = true;
            let mut flat_input_idx = 0usize;
            let mut stride_mult = 1usize;

            for d in (0..rank).rev() {
                let input_pos = out_idx[d] * strides[d] + win_idx[d];
                if input_pos >= input_dims[d] {
                    in_bounds = false;
                    break;
                }
                flat_input_idx += input_pos * stride_mult;
                stride_mult *= input_dims[d];
            }

            if in_bounds {
                let val = tensor.elements[flat_input_idx].as_f64().unwrap_or(0.0);
                match reduce_op {
                    "max" => accum = accum.max(val),
                    "min" => accum = accum.min(val),
                    _ => accum += val,
                }
            }

            // Increment window index
            let mut carry = true;
            for d in (0..rank).rev() {
                if carry {
                    win_idx[d] += 1;
                    if win_idx[d] >= window_dims[d] {
                        win_idx[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        output_elements.push(fj_core::Literal::from_f64(accum));

        // Increment output index
        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                out_idx[d] += 1;
                if out_idx[d] >= out_dims_usize[d] {
                    out_idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    Ok(Value::Tensor(
        TensorValue::new(
            fj_core::DType::F64,
            Shape { dims: out_dims },
            output_elements,
        )
        .map_err(EvalError::InvalidTensor)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::{EvalError, eval_primitive};
    use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
    use std::collections::BTreeMap;

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    fn pad_params(low: &str, high: &str, interior: &str) -> BTreeMap<String, String> {
        let mut params = BTreeMap::new();
        params.insert("padding_low".to_owned(), low.to_owned());
        params.insert("padding_high".to_owned(), high.to_owned());
        params.insert("padding_interior".to_owned(), interior.to_owned());
        params
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
    fn pad_vector_with_edge_padding() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let params = pad_params("1", "2", "0");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let expected = Value::vector_i64(&[0, 1, 2, 3, 0, 0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn pad_vector_with_interior_and_edge_padding() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let params = pad_params("1", "1", "1");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let expected = Value::vector_i64(&[0, 1, 0, 2, 0, 3, 0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn pad_rank2_tensor_preserves_layout() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let params = pad_params("1,0", "0,1", "0,1");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let out_tensor = out.as_tensor().expect("pad output should be tensor");
        assert_eq!(out_tensor.shape.dims, vec![3, 4]);
        assert_eq!(
            out_tensor.elements,
            vec![
                Literal::I64(0),
                Literal::I64(0),
                Literal::I64(0),
                Literal::I64(0),
                Literal::I64(1),
                Literal::I64(0),
                Literal::I64(2),
                Literal::I64(0),
                Literal::I64(3),
                Literal::I64(0),
                Literal::I64(4),
                Literal::I64(0),
            ]
        );
    }

    #[test]
    fn pad_rejects_negative_padding_values() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let params = pad_params("-1", "0", "0");
        let err =
            eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
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

    // ===================================================================
    // NaN / Inf edge cases (IEEE 754 compliance)
    // ===================================================================

    #[test]
    fn add_nan_propagates() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn sub_inf_inf_is_nan() {
        let out = eval_primitive(
            Primitive::Sub,
            &[
                Value::scalar_f64(f64::INFINITY),
                Value::scalar_f64(f64::INFINITY),
            ],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn mul_nan_propagates() {
        let out = eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(0.0)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn log_zero_is_neg_inf() {
        let out = eval_primitive(Primitive::Log, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v.is_infinite() && v < 0.0);
    }

    #[test]
    fn log_negative_is_nan() {
        let out = eval_primitive(Primitive::Log, &[Value::scalar_f64(-1.0)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let out =
            eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(-1.0)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn exp_neg_inf_is_zero() {
        let out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_f64(f64::NEG_INFINITY)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.0).abs() < 1e-15);
    }

    #[test]
    fn exp_inf_is_inf() {
        let out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_f64(f64::INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_infinite());
    }

    #[test]
    fn neg_zero_is_neg_zero() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v == 0.0 && v.is_sign_negative());
    }

    #[test]
    fn abs_neg_zero_is_pos_zero() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_f64(-0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v == 0.0 && v.is_sign_positive());
    }

    #[test]
    fn abs_nan_is_nan() {
        let out =
            eval_primitive(Primitive::Abs, &[Value::scalar_f64(f64::NAN)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn abs_neg_inf_is_inf() {
        let out = eval_primitive(
            Primitive::Abs,
            &[Value::scalar_f64(f64::NEG_INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::INFINITY);
    }

    #[test]
    fn reduce_sum_with_nan() {
        let input = Value::vector_f64(&[1.0, f64::NAN, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceSum, &[input], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn reduce_max_with_nan() {
        // f64::max returns the non-NaN value (Rust/JAX behavior)
        let input = Value::vector_f64(&[1.0, f64::NAN, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn max_nan_returns_other() {
        // f64::max(NaN, x) returns x (Rust/JAX behavior, not IEEE 754-2019 maximum)
        let out = eval_primitive(
            Primitive::Max,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(5.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn eq_nan_nan_is_false() {
        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn ne_nan_nan_is_true() {
        let out = eval_primitive(
            Primitive::Ne,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn lt_nan_always_false() {
        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    // ===================================================================
    // Type promotion tests
    // ===================================================================

    #[test]
    fn add_i64_f64_promotes_to_f64() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_f64(3.5)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 5.5).abs() < 1e-10);
    }

    #[test]
    fn sub_vector_broadcasts_scalar() {
        let vec = Value::vector_i64(&[10, 20, 30]).unwrap();
        let out =
            eval_primitive(Primitive::Sub, &[vec, Value::scalar_i64(5)], &no_params()).unwrap();
        let expected = Value::vector_i64(&[5, 15, 25]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn mul_scalar_broadcasts_to_vector() {
        let vec = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Mul, &[Value::scalar_i64(10), vec], &no_params()).unwrap();
        let expected = Value::vector_i64(&[10, 20, 30]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn pow_int_int_cast_through_float() {
        // pow(2, 10) should give 1024 (cast through f64)
        let out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_i64(2), Value::scalar_i64(10)],
            &no_params(),
        )
        .unwrap();
        // i64 pow goes through f64 path: (2 as f64).powf(10 as f64) as i64 = 1024
        if let Value::Scalar(fj_core::Literal::I64(v)) = out {
            assert_eq!(v, 1024);
        } else {
            panic!("expected i64 scalar from int pow");
        }
    }

    // ===================================================================
    // Broadcasting and shape mismatch error tests
    // ===================================================================

    #[test]
    fn add_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    #[test]
    fn binary_wrong_arity_error() {
        let err =
            eval_primitive(Primitive::Add, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn unary_wrong_arity_error() {
        let err = eval_primitive(
            Primitive::Neg,
            &[Value::scalar_i64(1), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn dot_wrong_arity_error() {
        let err =
            eval_primitive(Primitive::Dot, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn comparison_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    // ===================================================================
    // Reduction edge cases
    // ===================================================================

    #[test]
    fn reduce_sum_scalar_identity() {
        let out =
            eval_primitive(Primitive::ReduceSum, &[Value::scalar_i64(42)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(42));
    }

    #[test]
    fn reduce_prod_f64_vector() {
        let input = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 24.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_f64_with_neg_inf() {
        let input = Value::vector_f64(&[1.0, f64::NEG_INFINITY, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input], &no_params()).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn reduce_max_f64_with_inf() {
        let input = Value::vector_f64(&[1.0, f64::INFINITY, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::INFINITY);
    }

    // ===================================================================
    // Axis-aware reduction tests
    // ===================================================================

    fn axes_params(axes: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("axes".into(), axes.into());
        p
    }

    #[test]
    fn reduce_sum_axis0_rank2() {
        // [[1,2,3],[4,5,6]] shape [2,3] -> reduce axis 0 -> [5,7,9] shape [3]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0")).unwrap();
        let expected = Value::vector_i64(&[5, 7, 9]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_sum_axis1_rank2() {
        // [[1,2,3],[4,5,6]] shape [2,3] -> reduce axis 1 -> [6,15] shape [2]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("1")).unwrap();
        let expected = Value::vector_i64(&[6, 15]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_sum_both_axes_rank2() {
        // reducing both axes should give full reduction (scalar)
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0,1")).unwrap();
        assert_eq!(out, Value::scalar_i64(21));
    }

    #[test]
    fn reduce_max_axis0_rank2() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(5),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(2),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceMax, &[input], &axes_params("0")).unwrap();
        let expected = Value::vector_i64(&[4, 5, 6]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_min_axis1_rank2() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(3),
                    Literal::I64(1),
                    Literal::I64(5),
                    Literal::I64(6),
                    Literal::I64(2),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceMin, &[input], &axes_params("1")).unwrap();
        let expected = Value::vector_i64(&[1, 2]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_prod_axis0_rank2() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceProd, &[input], &axes_params("0")).unwrap();
        let expected = Value::vector_i64(&[4, 10, 18]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_sum_axis_out_of_bounds() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("1")).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn reduce_sum_axis0_rank3() {
        // shape [2,2,2] with values [1..8], reduce axis 0 -> shape [2,2]
        // [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                (1..=8).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let values: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(values, vec![6, 8, 10, 12]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn reduce_sum_f64_axis0() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3]);
            let values: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((values[0] - 5.0).abs() < 1e-10);
            assert!((values[1] - 7.0).abs() < 1e-10);
            assert!((values[2] - 9.0).abs() < 1e-10);
        } else {
            panic!("expected tensor output");
        }
    }

    // ===================================================================
    // Tensor manipulation edge cases
    // ===================================================================

    #[test]
    fn reshape_incompatible_size_error() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,2".into());
        let err = eval_primitive(Primitive::Reshape, &[input], &params).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    #[test]
    fn transpose_invalid_permutation_error() {
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
        let mut params = BTreeMap::new();
        params.insert("permutation".into(), "0".into()); // wrong length
        let err =
            eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn transpose_scalar_is_identity() {
        let out =
            eval_primitive(Primitive::Transpose, &[Value::scalar_i64(42)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(42));
    }

    #[test]
    fn slice_out_of_bounds_error() {
        let input = Value::vector_i64(&[10, 20, 30]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "5".into()); // exceeds dim
        let err = eval_primitive(Primitive::Slice, &[input], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn concatenate_scalar_error() {
        let err = eval_primitive(
            Primitive::Concatenate,
            &[Value::scalar_i64(1), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn gather_wrong_arity_rejected() {
        let err =
            eval_primitive(Primitive::Gather, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ArityMismatch { .. }));
    }

    #[test]
    fn gather_scalar_operand_rejected() {
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(0)]).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());
        let err = eval_primitive(Primitive::Gather, &[Value::scalar_i64(1), indices], &params)
            .unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn gather_1d_indices_from_2d() {
        // operand: [[10,20],[30,40],[50,60]] (shape [3,2])
        // indices: [2, 0] — gather rows 2 and 0
        // slice_sizes: 1,2
        // result: [[50,60],[10,20]] (shape [2,2])
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                    Literal::I64(50),
                    Literal::I64(60),
                ],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(2), Literal::I64(0)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,2".into());

        let out = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        panic!()
                    }
                })
                .collect();
            assert_eq!(vals, vec![50, 60, 10, 20]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn gather_out_of_bounds_rejected() {
        let operand = Value::vector_i64(&[1, 2, 3]).unwrap();
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(5)]).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());
        let err = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn scatter_1d_indices_into_2d() {
        // operand: [[0,0],[0,0],[0,0]] (shape [3,2])
        // indices: [1, 0]
        // updates: [[10,20],[30,40]] (shape [2,2])
        // result:  [[30,40],[10,20],[0,0]]
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                vec![Literal::I64(0); 6],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(1), Literal::I64(0)],
            )
            .unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        panic!()
                    }
                })
                .collect();
            assert_eq!(vals, vec![30, 40, 10, 20, 0, 0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn scatter_scalar_operand_rejected() {
        let err =
            eval_primitive(Primitive::Scatter, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ArityMismatch { .. }));
    }

    #[test]
    fn gather_1d_simple() {
        // operand: [10, 20, 30, 40, 50] (shape [5])
        // indices: [3, 1, 4]
        // slice_sizes: 1
        // result: [40, 20, 50]
        let operand = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(3),
                vec![Literal::I64(3), Literal::I64(1), Literal::I64(4)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());

        let out = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        panic!()
                    }
                })
                .collect();
            assert_eq!(vals, vec![40, 20, 50]);
        } else {
            panic!("expected tensor");
        }
    }

    // ===================================================================
    // Trigonometric tests
    // ===================================================================

    #[test]
    fn sin_zero_is_zero() {
        let out = eval_primitive(Primitive::Sin, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.0).abs() < 1e-15);
    }

    #[test]
    fn cos_zero_is_one() {
        let out = eval_primitive(Primitive::Cos, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sin_pi_is_zero() {
        let out = eval_primitive(
            Primitive::Sin,
            &[Value::scalar_f64(std::f64::consts::PI)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn sin_nan_is_nan() {
        let out =
            eval_primitive(Primitive::Sin, &[Value::scalar_f64(f64::NAN)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    // ===================================================================
    // Comparison with broadcast (scalar + tensor)
    // ===================================================================

    #[test]
    fn gt_scalar_tensor_broadcast() {
        let vec = Value::vector_i64(&[1, 5, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Gt, &[Value::scalar_i64(3), vec], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true)); // 3 > 1
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false)); // 3 > 5
            assert_eq!(t.elements[2], fj_core::Literal::Bool(false)); // 3 > 3
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn le_tensor_scalar_broadcast() {
        let vec = Value::vector_i64(&[1, 5, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Le, &[vec, Value::scalar_i64(3)], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true)); // 1 <= 3
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false)); // 5 <= 3
            assert_eq!(t.elements[2], fj_core::Literal::Bool(true)); // 3 <= 3
        } else {
            panic!("expected tensor");
        }
    }

    // ===================================================================
    // Dot product edge cases
    // ===================================================================

    #[test]
    fn dot_f64_vectors() {
        let lhs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let rhs = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
        let out = eval_primitive(Primitive::Dot, &[lhs, rhs], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 32.0).abs() < 1e-10);
    }

    #[test]
    fn dot_scalar_multiply() {
        let out = eval_primitive(
            Primitive::Dot,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(21));
    }

    #[test]
    fn dot_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    // ===================================================================
    // EvalError display formatting
    // ===================================================================

    #[test]
    fn eval_error_display_formatting() {
        let err = EvalError::ArityMismatch {
            primitive: Primitive::Add,
            expected: 2,
            actual: 1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("add"));
        assert!(msg.contains("expected 2"));
        assert!(msg.contains("got 1"));
    }

    // ===================================================================
    // Vector-level elementwise tests for transcendentals
    // ===================================================================

    #[test]
    fn exp_vector() {
        let input = Value::vector_f64(&[0.0, 1.0]).unwrap();
        let out = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let e0 = t.elements[0].as_f64().unwrap();
            let e1 = t.elements[1].as_f64().unwrap();
            assert!((e0 - 1.0).abs() < 1e-10);
            assert!((e1 - std::f64::consts::E).abs() < 1e-10);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn floor_vector() {
        let input = Value::vector_f64(&[1.9, 2.1, -0.5]).unwrap();
        let out = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert!((t.elements[0].as_f64().unwrap() - 1.0).abs() < 1e-10);
            assert!((t.elements[1].as_f64().unwrap() - 2.0).abs() < 1e-10);
            assert!((t.elements[2].as_f64().unwrap() - (-1.0)).abs() < 1e-10);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn rsqrt_vector() {
        let input = Value::vector_f64(&[1.0, 4.0, 16.0]).unwrap();
        let out = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert!((t.elements[0].as_f64().unwrap() - 1.0).abs() < 1e-10);
            assert!((t.elements[1].as_f64().unwrap() - 0.5).abs() < 1e-10);
            assert!((t.elements[2].as_f64().unwrap() - 0.25).abs() < 1e-10);
        } else {
            panic!("expected tensor");
        }
    }

    // ── Select broadcasting tests ─────────────────────────────────

    #[test]
    fn select_scalar_all_scalars() {
        let out = eval_primitive(
            Primitive::Select,
            &[
                Value::scalar_bool(true),
                Value::scalar_f64(1.0),
                Value::scalar_f64(0.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::Scalar(Literal::from_f64(1.0)));
    }

    #[test]
    fn select_tensor_cond_scalar_values() {
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape::vector(3),
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::Select,
            &[cond, Value::scalar_f64(10.0), Value::scalar_f64(-1.0)],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![10.0, -1.0, 10.0]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn select_scalar_cond_tensor_values() {
        let on_true = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let on_false = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
        let out = eval_primitive(
            Primitive::Select,
            &[Value::scalar_bool(false), on_true, on_false.clone()],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, on_false);
    }

    // ===================================================================
    // Scatter mode="add" tests
    // ===================================================================

    #[test]
    fn scatter_add_mode_accumulates() {
        // operand: [0.0, 0.0, 0.0] (shape [3])
        // indices: [1, 1]  (duplicate index)
        // updates: [10.0, 20.0]
        // With mode="add", index 1 should accumulate: 0 + 10 + 20 = 30
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(0.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(0.0),
                ],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(1), Literal::I64(1)],
            )
            .unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(2),
                vec![Literal::from_f64(10.0), Literal::from_f64(20.0)],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("mode".into(), "add".into());

        let out =
            eval_primitive(Primitive::Scatter, &[operand, indices, updates], &params).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals[0], 0.0);
            assert_eq!(vals[1], 30.0); // 10 + 20 accumulated
            assert_eq!(vals[2], 0.0);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn scatter_add_preserves_existing_values() {
        // operand: [100.0, 200.0, 300.0]
        // indices: [0]
        // updates: [5.0]
        // mode="add": result[0] = 100 + 5 = 105
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(100.0),
                    Literal::from_f64(200.0),
                    Literal::from_f64(300.0),
                ],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(0)]).unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(DType::F64, Shape::vector(1), vec![Literal::from_f64(5.0)]).unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("mode".into(), "add".into());

        let out =
            eval_primitive(Primitive::Scatter, &[operand, indices, updates], &params).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![105.0, 200.0, 300.0]);
        } else {
            panic!("expected tensor");
        }
    }

    // ===================================================================
    // Concatenate edge cases
    // ===================================================================

    // ===================================================================
    // Gather edge cases
    // ===================================================================

    #[test]
    fn gather_slice_sizes_exceed_operand_dims_rejected() {
        // operand shape [3, 2], slice_sizes [1, 5] — 5 > 2 should fail
        let operand = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::F64,
                fj_core::Shape { dims: vec![3, 2] },
                vec![
                    fj_core::Literal::from_f64(1.0),
                    fj_core::Literal::from_f64(2.0),
                    fj_core::Literal::from_f64(3.0),
                    fj_core::Literal::from_f64(4.0),
                    fj_core::Literal::from_f64(5.0),
                    fj_core::Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let indices = Value::scalar_i64(0);
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,5".into());
        let result = eval_primitive(Primitive::Gather, &[operand, indices], &params);
        assert!(result.is_err(), "slice_sizes[1]=5 exceeds dim=2");
    }

    #[test]
    fn gather_empty_indices() {
        let operand = Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap();
        let indices = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::I64,
                fj_core::Shape { dims: vec![0] },
                vec![],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());
        let result = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.elements.len(), 0);
            assert_eq!(t.shape.dims[0], 0);
        } else {
            panic!("expected tensor");
        }
    }

    // ===================================================================
    // Scatter edge cases
    // ===================================================================

    #[test]
    fn scatter_duplicate_indices_overwrite_last_wins() {
        // indices [0, 0] with updates [10, 20] — last write wins
        let operand = Value::vector_f64(&[0.0, 0.0, 0.0]).unwrap();
        let indices = Value::vector_i64(&[0, 0]).unwrap();
        let updates = Value::vector_f64(&[10.0, 20.0]).unwrap();
        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals[0], 20.0, "last write should win for duplicate indices");
            assert_eq!(vals[1], 0.0);
            assert_eq!(vals[2], 0.0);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn scatter_unknown_mode_rejected() {
        let operand = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let indices = Value::vector_i64(&[0]).unwrap();
        let updates = Value::vector_f64(&[9.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("mode".into(), "invalid_mode".into());
        let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &params);
        assert!(result.is_err(), "unknown mode should be rejected");
    }

    #[test]
    fn scatter_updates_shape_mismatch_rejected() {
        // operand [3], indices [1], updates [2] — updates has 2 elems but expected 1
        let operand = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let indices = Value::vector_i64(&[0]).unwrap();
        let updates = Value::vector_f64(&[10.0, 20.0]).unwrap();
        let result = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        );
        assert!(
            result.is_err(),
            "updates element count mismatch should error"
        );
    }

    // ===================================================================
    // Slice edge cases
    // ===================================================================

    #[test]
    fn slice_empty_result() {
        // slice with start == limit should produce empty tensor
        let v = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "2".into());
        params.insert("limit_indices".into(), "2".into());
        let out = eval_primitive(Primitive::Slice, &[v], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 0);
            assert_eq!(t.shape.dims[0], 0);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn slice_single_element() {
        let v = Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "2".into());
        let out = eval_primitive(Primitive::Slice, &[v], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 1);
            assert_eq!(t.elements[0].as_f64().unwrap(), 20.0);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn slice_start_exceeds_limit_rejected() {
        let v = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "2".into());
        params.insert("limit_indices".into(), "1".into());
        let result = eval_primitive(Primitive::Slice, &[v], &params);
        assert!(result.is_err(), "start > limit should error");
    }

    #[test]
    fn concatenate_single_input() {
        let a = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out = eval_primitive(
            Primitive::Concatenate,
            std::slice::from_ref(&a),
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, a);
    }

    #[test]
    fn concatenate_three_inputs() {
        let a = Value::vector_i64(&[1]).unwrap();
        let b = Value::vector_i64(&[2, 3]).unwrap();
        let c = Value::vector_i64(&[4, 5, 6]).unwrap();
        let out = eval_primitive(Primitive::Concatenate, &[a, b, c], &no_params()).unwrap();
        let expected = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(out, expected);
    }

    // ══════════════════════════════════════════════════════════════
    // Clamp tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn clamp_scalar_within_range() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_f64(3.0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(5.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(3.0));
    }

    #[test]
    fn clamp_scalar_below_min() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_f64(-2.0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(5.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(1.0));
    }

    #[test]
    fn clamp_scalar_above_max() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_f64(10.0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(5.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(5.0));
    }

    #[test]
    fn clamp_i64_scalar() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_i64(10),
                Value::scalar_i64(0),
                Value::scalar_i64(5),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(5));
    }

    #[test]
    fn clamp_tensor_with_scalar_bounds() {
        let x = Value::vector_f64(&[-1.0, 2.0, 5.0, 8.0]).unwrap();
        let out = eval_primitive(
            Primitive::Clamp,
            &[x, Value::scalar_f64(0.0), Value::scalar_f64(6.0)],
            &no_params(),
        )
        .unwrap();
        let expected = Value::vector_f64(&[0.0, 2.0, 5.0, 6.0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn clamp_arity_error() {
        let result = eval_primitive(
            Primitive::Clamp,
            &[Value::scalar_f64(1.0), Value::scalar_f64(0.0)],
            &no_params(),
        );
        assert!(result.is_err());
    }

    // ══════════════════════════════════════════════════════════════
    // Iota tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn iota_i64_length_5() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "5".to_owned());
        let out = eval_primitive(Primitive::Iota, &[], &params).unwrap();
        let expected = Value::vector_i64(&[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn iota_f64() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "3".to_owned());
        params.insert("dtype".to_owned(), "F64".to_owned());
        let out = eval_primitive(Primitive::Iota, &[], &params).unwrap();
        let expected = Value::vector_f64(&[0.0, 1.0, 2.0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn iota_zero_length() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "0".to_owned());
        let out = eval_primitive(Primitive::Iota, &[], &params).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, Shape::vector(0));
                assert!(t.elements.is_empty());
            }
            _ => panic!("expected tensor"),
        }
    }

    #[test]
    fn iota_arity_error_with_input() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "3".to_owned());
        let result = eval_primitive(Primitive::Iota, &[Value::scalar_i64(1)], &params);
        assert!(result.is_err());
    }

    // ══════════════════════════════════════════════════════════════
    // DynamicSlice tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn dynamic_slice_1d() {
        let x = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "3".to_owned());
        let out =
            eval_primitive(Primitive::DynamicSlice, &[x, Value::scalar_i64(1)], &params).unwrap();
        let expected = Value::vector_i64(&[20, 30, 40]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dynamic_slice_start_clamping() {
        // JAX clamps start indices to valid range
        let x = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "3".to_owned());
        // Start index 10 should be clamped to 2 (5 - 3 = 2)
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(10)],
            &params,
        )
        .unwrap();
        let expected = Value::vector_i64(&[30, 40, 50]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dynamic_slice_negative_start_clamped() {
        let x = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2".to_owned());
        // Negative start should be clamped to 0
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(-5)],
            &params,
        )
        .unwrap();
        let expected = Value::vector_i64(&[10, 20]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dynamic_slice_2d() {
        let t = TensorValue::new(
            DType::I64,
            Shape { dims: vec![3, 4] },
            (0..12).map(Literal::I64).collect(),
        )
        .unwrap();
        let x = Value::Tensor(t);
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2,2".to_owned());
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(1), Value::scalar_i64(1)],
            &params,
        )
        .unwrap();
        // Extracting a 2x2 block starting at (1,1) from a 3x4 matrix
        // Matrix: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        // Expected: [[5,6],[9,10]]
        let expected = TensorValue::new(
            DType::I64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::I64(5),
                Literal::I64(6),
                Literal::I64(9),
                Literal::I64(10),
            ],
        )
        .unwrap();
        assert_eq!(out, Value::Tensor(expected));
    }

    #[test]
    fn dynamic_slice_scalar_error() {
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "1".to_owned());
        let result = eval_primitive(
            Primitive::DynamicSlice,
            &[Value::scalar_i64(42), Value::scalar_i64(0)],
            &params,
        );
        assert!(result.is_err());
    }

    // ── Higher-rank gather/scatter tests ─────────────────────────

    #[test]
    fn gather_rank3_operand() {
        // operand: shape [3, 2, 2] — 3 matrices of 2x2
        // indices: [2, 0]
        // slice_sizes: 1, 2, 2
        // result: shape [2, 2, 2] — matrices 2 and 0
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![3, 2, 2],
                },
                (1..=12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(2), Literal::I64(0)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,2,2".into());

        let out = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        panic!()
                    }
                })
                .collect();
            // index 2: elements 9,10,11,12; index 0: elements 1,2,3,4
            assert_eq!(vals, vec![9, 10, 11, 12, 1, 2, 3, 4]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn scatter_into_rank3_operand() {
        // operand: shape [3, 2, 2] all zeros
        // indices: [1]
        // updates: shape [1, 2, 2] = [[10, 20], [30, 40]]
        // result: slot 1 of operand replaced with updates
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![3, 2, 2],
                },
                vec![Literal::I64(0); 12],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(1)]).unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![1, 2, 2],
                },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 2, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        panic!()
                    }
                })
                .collect();
            // Slot 0: [0,0,0,0], Slot 1: [10,20,30,40], Slot 2: [0,0,0,0]
            assert_eq!(vals, vec![0, 0, 0, 0, 10, 20, 30, 40, 0, 0, 0, 0]);
        } else {
            panic!("expected tensor");
        }
    }

    // ── Select tensor tests ────────────────────────────────────────────

    #[test]
    fn select_tensor_condition_picks_elementwise() {
        // cond=[true,false,true], on_true=[10,20,30], on_false=[1,2,3]
        // expected: [10,2,30]
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape::vector(3),
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let on_true = Value::vector_i64(&[10, 20, 30]).unwrap();
        let on_false = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![10, 2, 30]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn select_rank2_tensor() {
        // 2x2 tensors
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let on_true = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let on_false = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );
        let out =
            eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 20, 30, 4]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn select_tensor_shape_mismatch_errors() {
        let cond = Value::Tensor(
            TensorValue::new(DType::Bool, Shape::vector(3), vec![Literal::Bool(true); 3]).unwrap(),
        );
        let on_true = Value::vector_i64(&[1, 2]).unwrap(); // shape [2] != [3]
        let on_false = Value::vector_i64(&[10, 20]).unwrap();
        let result = eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params());
        assert!(result.is_err());
    }

    // ── OneHot tests ──────────────────────────────────────────────────

    fn one_hot_params(num_classes: u32) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("num_classes".to_owned(), num_classes.to_string());
        p
    }

    #[test]
    fn one_hot_vector_indices() {
        // one_hot([0, 2, 1], num_classes=3) → [[1,0,0],[0,0,1],[0,1,0]]
        let indices = Value::vector_i64(&[0, 2, 1]).unwrap();
        let out = eval_primitive(Primitive::OneHot, &[indices], &one_hot_params(3)).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 3]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn one_hot_scalar_index() {
        // one_hot(2, num_classes=4) → [0,0,1,0]
        let indices = Value::scalar_i64(2);
        let out = eval_primitive(Primitive::OneHot, &[indices], &one_hot_params(4)).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![4]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![0.0, 0.0, 1.0, 0.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn one_hot_out_of_range_index() {
        // Negative index → all off_value
        let indices = Value::scalar_i64(-1);
        let out = eval_primitive(Primitive::OneHot, &[indices], &one_hot_params(3)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![0.0, 0.0, 0.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn one_hot_custom_on_off_values() {
        let mut p = one_hot_params(3);
        p.insert("on_value".to_owned(), "5.0".to_owned());
        p.insert("off_value".to_owned(), "-1.0".to_owned());
        let indices = Value::scalar_i64(1);
        let out = eval_primitive(Primitive::OneHot, &[indices], &p).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![-1.0, 5.0, -1.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn one_hot_missing_num_classes_errors() {
        let indices = Value::scalar_i64(0);
        let result = eval_primitive(Primitive::OneHot, &[indices], &no_params());
        assert!(result.is_err());
    }

    // ── DynamicUpdateSlice tests ──────────────────────────────────────

    #[test]
    fn dynamic_update_slice_1d() {
        // operand: [0, 0, 0, 0, 0], update: [10, 20], start: 2
        // result: [0, 0, 10, 20, 0]
        let operand = Value::vector_i64(&[0, 0, 0, 0, 0]).unwrap();
        let update = Value::vector_i64(&[10, 20]).unwrap();
        let start = Value::scalar_i64(2);
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![5]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![0, 0, 10, 20, 0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_2d() {
        // operand: [[0,0,0],[0,0,0]], update: [[7,8]], start: (1,1)
        // result: [[0,0,0],[0,7,8]]
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![Literal::I64(0); 6],
            )
            .unwrap(),
        );
        let update = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![1, 2] },
                vec![Literal::I64(7), Literal::I64(8)],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_i64(1), Value::scalar_i64(1)],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![0, 0, 0, 0, 7, 8]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_clamped_start() {
        // Start index out of range should be clamped
        // operand: [1, 2, 3], update: [99, 88], start: 10 → clamped to 1
        let operand = Value::vector_i64(&[1, 2, 3]).unwrap();
        let update = Value::vector_i64(&[99, 88]).unwrap();
        let start = Value::scalar_i64(10);
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 99, 88]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_arity_error() {
        let operand = Value::vector_i64(&[1, 2]).unwrap();
        let result = eval_primitive(Primitive::DynamicUpdateSlice, &[operand], &no_params());
        assert!(result.is_err());
    }

    // ── Cumsum / Cumprod tests ────────────────────────────────────────

    fn axis_params(axis: usize) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("axis".to_owned(), axis.to_string());
        p
    }

    #[test]
    fn cumsum_1d() {
        // cumsum([1, 2, 3, 4]) = [1, 3, 6, 10]
        let input = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 3, 6, 10]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn cumsum_2d_axis0() {
        // [[1, 2], [3, 4]] cumsum axis=0 → [[1, 2], [4, 6]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Cumsum, &[input], &axis_params(0)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 2, 4, 6]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn cumsum_2d_axis1() {
        // [[1, 2], [3, 4]] cumsum axis=1 → [[1, 3], [3, 7]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Cumsum, &[input], &axis_params(1)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 3, 3, 7]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn cumprod_1d() {
        // cumprod([1, 2, 3, 4]) = [1, 2, 6, 24]
        let input = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 2, 6, 24]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn cumprod_f64() {
        // cumprod([1.0, 2.0, 3.0]) = [1.0, 2.0, 6.0]
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let out = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 2.0, 6.0]);
        } else {
            panic!("expected tensor");
        }
    }

    // ── Sort / Argsort tests ──────────────────────────────────────────

    #[test]
    fn sort_1d_ascending() {
        let input = Value::vector_i64(&[3, 1, 4, 1, 5]).unwrap();
        let out = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 1, 3, 4, 5]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn sort_1d_descending() {
        let mut p = BTreeMap::new();
        p.insert("descending".to_owned(), "true".to_owned());
        let input = Value::vector_i64(&[3, 1, 4, 1, 5]).unwrap();
        let out = eval_primitive(Primitive::Sort, &[input], &p).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![5, 4, 3, 1, 1]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn argsort_1d() {
        let input = Value::vector_i64(&[30, 10, 20]).unwrap();
        let out = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 2, 0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn sort_2d_axis1() {
        // [[3, 1], [4, 2]] sorted along axis 1 → [[1, 3], [2, 4]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(3),
                    Literal::I64(1),
                    Literal::I64(4),
                    Literal::I64(2),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Sort, &[input], &axis_params(1)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 3, 2, 4]);
        } else {
            panic!("expected tensor");
        }
    }

    // ── Conv tests ────────────────────────────────────────────────────

    fn conv_params(padding: &str, strides: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("padding".to_owned(), padding.to_owned());
        p.insert("strides".to_owned(), strides.to_owned());
        p
    }

    #[test]
    fn conv_1d_valid_single_channel() {
        // lhs=[1, 4, 1] (batch=1, width=4, channels=1)
        // rhs=[2, 1, 1] (kernel=2, c_in=1, c_out=1)
        // valid padding, stride=1 → output=[1, 3, 1]
        // input: [1, 2, 3, 4], kernel: [1, 1]
        // out: [1*1+2*1, 2*1+3*1, 3*1+4*1] = [3, 5, 7]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 4, 1],
                },
                vec![1.0, 2.0, 3.0, 4.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![1.0, 1.0].into_iter().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 5.0, 7.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn conv_1d_same_padding() {
        // lhs=[1, 3, 1], rhs=[3, 1, 1], same padding, stride=1
        // input: [1, 2, 3], kernel: [1, 1, 1]
        // same → output width=3
        // pad_left=1: padded=[0, 1, 2, 3, 0]
        // out: [0+1+2, 1+2+3, 2+3+0] = [3, 6, 5]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![1.0, 2.0, 3.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![3, 1, 1],
                },
                vec![1.0, 1.0, 1.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 6.0, 5.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn conv_1d_stride2() {
        // lhs=[1, 6, 1], rhs=[2, 1, 1], valid, stride=2
        // input: [1,2,3,4,5,6], kernel: [1,1]
        // output width = (6-2)/2+1 = 3
        // positions: 0,2,4 → [1+2, 3+4, 5+6] = [3, 7, 11]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 6, 1],
                },
                (1..=6).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "2")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 7.0, 11.0]);
        } else {
            panic!("expected tensor");
        }
    }

    // ── Conv 2D tests ────────────────────────────────────────────

    #[test]
    fn conv_2d_valid_single_channel() {
        // lhs=[1, 3, 3, 1], rhs=[2, 2, 1, 1], valid, stride=1
        // Input 3x3 image:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        // Kernel 2x2 (all ones):
        // 1 1
        // 1 1
        // Output 2x2: [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] = [12, 16, 24, 28]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 3, 1],
                },
                (1..=9).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 1, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 2, 2, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![12.0, 16.0, 24.0, 28.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn conv_2d_same_padding() {
        // lhs=[1, 3, 3, 1], rhs=[3, 3, 1, 1], same padding
        // With same padding, output should have same spatial dims as input: 3x3
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 3, 1],
                },
                (1..=9).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![3, 3, 1, 1],
                },
                vec![Literal::from_f64(1.0); 9],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 3, 1]);
            // Center element: sum of all 9 values = 45
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((vals[4] - 45.0).abs() < 1e-10, "center = {}", vals[4]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn conv_2d_multi_channel() {
        // lhs=[1, 2, 2, 2] (2x2 image, 2 channels)
        // rhs=[1, 1, 2, 3] (1x1 kernel, 2 c_in, 3 c_out) -- pointwise conv
        // This is effectively a dense transform of each pixel's channel vector
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 2, 2, 2],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0), // pixel (0,0): [1,2]
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0), // pixel (0,1): [3,4]
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0), // pixel (1,0): [5,6]
                    Literal::from_f64(7.0),
                    Literal::from_f64(8.0), // pixel (1,1): [7,8]
                ],
            )
            .unwrap(),
        );
        // kernel: 1x1, c_in=2, c_out=3
        // rhs layout: [KH=1, KW=1, C_in=2, C_out=3]
        // W = [[1,0,1], [0,1,1]] -> output channels: ch0=ci0, ch1=ci1, ch2=ci0+ci1
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 1, 2, 3],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(1.0), // ci=0: [1,0,1]
                    Literal::from_f64(0.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0), // ci=1: [0,1,1]
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 2, 2, 3]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            // pixel (0,0): [1,2] -> [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
            assert_eq!(vals[0..3], [1.0, 2.0, 3.0]);
            // pixel (0,1): [3,4] -> [3, 4, 7]
            assert_eq!(vals[3..6], [3.0, 4.0, 7.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn cond_true_returns_true_branch() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::scalar_f64(42.0);
        let false_val = Value::scalar_f64(99.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }

    #[test]
    fn cond_false_returns_false_branch() {
        let pred = Value::scalar_bool(false);
        let true_val = Value::scalar_f64(42.0);
        let false_val = Value::scalar_f64(99.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 99.0);
    }

    #[test]
    fn cond_with_tensor_branches() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let false_val = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        if let Value::Tensor(t) = out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn cond_i64_pred_nonzero_is_true() {
        let pred = Value::scalar_i64(1);
        let true_val = Value::scalar_f64(10.0);
        let false_val = Value::scalar_f64(20.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    #[test]
    fn cond_arity_error() {
        let params = BTreeMap::new();
        let result = eval_primitive(Primitive::Cond, &[Value::scalar_bool(true)], &params);
        assert!(result.is_err());
    }

    // ── Scan tests ──────────────────────────────────────────────────

    fn scan_params(body_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p
    }

    fn scan_params_reverse(body_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p.insert("reverse".to_owned(), "true".to_owned());
        p
    }

    #[test]
    fn scan_add_vector() {
        // scan(add, 0.0, [1,2,3,4]) => 0+1+2+3+4 = 10
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    #[test]
    fn scan_mul_vector() {
        // scan(mul, 1.0, [2,3,4]) => 1*2*3*4 = 24
        let init = Value::scalar_f64(1.0);
        let xs = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("mul")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 24.0);
    }

    #[test]
    fn scan_add_reverse() {
        // scan(add, 0.0, [1,2,3], reverse=true) => 0+3+2+1 = 6 (same as forward for add)
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let out =
            eval_primitive(Primitive::Scan, &[init, xs], &scan_params_reverse("add")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 6.0);
    }

    #[test]
    fn scan_sub_reverse() {
        // scan(sub, 10.0, [1,2,3], reverse=true) => ((10-3)-2)-1 = 4
        let init = Value::scalar_f64(10.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let out =
            eval_primitive(Primitive::Scan, &[init, xs], &scan_params_reverse("sub")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 4.0);
    }

    #[test]
    fn scan_max_vector() {
        // scan(max, -inf, [3,1,4,1,5]) => 5
        let init = Value::scalar_f64(f64::NEG_INFINITY);
        let xs = Value::vector_f64(&[3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("max")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 5.0);
    }

    #[test]
    fn scan_with_tensor_slices() {
        // xs shape [2, 3]: scan(add, [0,0,0], [[1,2,3],[4,5,6]]) => [5,7,9]
        let init = Value::vector_f64(&[0.0, 0.0, 0.0]).unwrap();
        let xs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
        if let Value::Tensor(t) = out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![5.0, 7.0, 9.0]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn scan_arity_error() {
        let result = eval_primitive(Primitive::Scan, &[Value::scalar_f64(0.0)], &no_params());
        assert!(result.is_err());
    }

    #[test]
    fn scan_empty_tensor() {
        // Scan over empty leading axis returns init_carry unchanged
        let init = Value::scalar_f64(42.0);
        let xs =
            Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap());
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }

    // ── Scan functional tests (bd-3eyv) ──────────────────────────────

    #[test]
    fn test_scan_accumulate_sum() {
        // scan(add, init=0, xs=[1,2,3,4]) → carry=10, ys=[1,3,6,10]
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 10.0);
        assert_eq!(ys.len(), 1);
        let ys_tensor = ys[0].as_tensor().expect("ys should be tensor");
        let ys_vals = ys_tensor.to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_scan_accumulate_product() {
        // scan(mul, init=1, xs=[1,2,3,4]) → carry=24, ys=[1,2,6,24]
        let init = vec![Value::scalar_f64(1.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Mul, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 24.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_scan_custom_body() {
        // Custom body: carry = carry + x * 2
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let doubled = eval_primitive(
                    Primitive::Mul,
                    &[x, Value::scalar_f64(2.0)],
                    &BTreeMap::new(),
                )?;
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), doubled], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        // 0 + 1*2 = 2, 2 + 2*2 = 6, 6 + 3*2 = 12
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 12.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_scan_multi_carry() {
        // Multi-carry: (count, sum). Body: count += 1, sum += x
        let init = vec![Value::scalar_f64(0.0), Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_count = eval_primitive(
                    Primitive::Add,
                    &[c[0].clone(), Value::scalar_f64(1.0)],
                    &BTreeMap::new(),
                )?;
                let new_sum =
                    eval_primitive(Primitive::Add, &[c[1].clone(), x.clone()], &BTreeMap::new())?;
                Ok((vec![new_count, new_sum], vec![x]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 3.0); // count
        assert_eq!(carry[1].as_f64_scalar().unwrap(), 60.0); // sum
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![10.0, 20.0, 30.0]); // identity output
    }

    #[test]
    fn test_scan_no_output() {
        // Scan that only accumulates carry, no per-step output
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry], vec![]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 6.0);
        assert!(ys.is_empty());
    }

    #[test]
    fn test_scan_empty_xs() {
        // Scan over empty array → carry = init, ys = empty
        let init = vec![Value::scalar_f64(42.0)];
        let xs =
            Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap());
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 42.0);
        assert!(ys.is_empty());
    }

    #[test]
    fn test_scan_single_element() {
        // Scan over single-element array
        let init = vec![Value::scalar_f64(5.0)];
        let xs = Value::vector_f64(&[3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 8.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![8.0]);
    }

    #[test]
    fn test_scan_tensor_carry() {
        // Scan with rank-2 tensor as carry
        // carry is [2] vector, xs is [3] vector, body: carry = carry + [x, x]
        let init = vec![Value::vector_f64(&[0.0, 0.0]).unwrap()];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, _ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                // Build a [2] tensor from the scalar x
                let x_val = x.as_f64_scalar().unwrap();
                let x_vec = Value::vector_f64(&[x_val, x_val]).unwrap();
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x_vec], &BTreeMap::new())?;
                Ok((vec![new_carry], vec![]))
            },
            false,
        )
        .unwrap();
        // carry should be [0+1+2+3, 0+1+2+3] = [6, 6]
        let carry_tensor = carry[0].as_tensor().unwrap();
        let carry_vals = carry_tensor.to_f64_vec().unwrap();
        assert_eq!(carry_vals, vec![6.0, 6.0]);
    }

    #[test]
    fn test_scan_reverse() {
        // scan(add, init=0, xs=[1,2,3], reverse=true) → iterate 3,2,1
        // ys=[3, 5, 6] (cumsum from reverse)
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            true,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 6.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        // Reverse: 0+3=3, 3+2=5, 5+1=6
        assert_eq!(ys_vals, vec![3.0, 5.0, 6.0]);
    }

    // ── While loop tests ────────────────────────────────────────────

    fn while_params(body_op: &str, cond_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p.insert("cond_op".to_owned(), cond_op.to_owned());
        p
    }

    fn while_params_max(body_op: &str, cond_op: &str, max: usize) -> BTreeMap<String, String> {
        let mut p = while_params(body_op, cond_op);
        p.insert("max_iter".to_owned(), max.to_string());
        p
    }

    #[test]
    fn while_add_until_ge_threshold() {
        // while carry < 10: carry += 3 => 0, 3, 6, 9, 12 => stops at 12
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(3.0);
        let threshold = Value::scalar_f64(10.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 12.0);
    }

    #[test]
    fn while_mul_until_ge_threshold() {
        // while carry < 100: carry *= 2 => 1, 2, 4, 8, 16, 32, 64, 128 => stops at 128
        let init = Value::scalar_f64(1.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(100.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("mul", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 128.0);
    }

    #[test]
    fn while_sub_until_le_zero() {
        // while carry > 0: carry -= 2 => 10, 8, 6, 4, 2, 0 => stops at 0
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(0.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("sub", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn while_condition_false_immediately() {
        // carry = 10, while carry < 5: carry += 1 => condition false, returns 10
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(5.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    #[test]
    fn while_arity_error() {
        let result = eval_primitive(
            Primitive::While,
            &[Value::scalar_f64(0.0)],
            &while_params("add", "lt"),
        );
        assert!(result.is_err());
    }

    // ── New while_loop tests (bd-2807) ─────────────────────────────

    #[test]
    fn test_while_loop_countdown() {
        // while_loop(|x| x > 0, |x| x - 1, init=10) → 0
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(0.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("sub", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn test_while_loop_convergence() {
        // Newton's method for sqrt(2): x_{n+1} = (x + 2/x) / 2
        // We approximate by iterating: carry = (carry + 2/carry) / 2
        // Using the functional API
        let init = vec![Value::scalar_f64(2.0)];
        let result = super::eval_while_loop_functional(
            init,
            100,
            |carry| {
                let x = carry[0].as_f64_scalar().unwrap();
                // Continue while |x² - 2| > 1e-10
                Ok((x * x - 2.0).abs() > 1e-10)
            },
            |carry| {
                let x = carry[0].as_f64_scalar().unwrap();
                let new_x = (x + 2.0 / x) / 2.0;
                Ok(vec![Value::scalar_f64(new_x)])
            },
        )
        .unwrap();
        let sqrt2 = result[0].as_f64_scalar().unwrap();
        assert!(
            (sqrt2 - std::f64::consts::SQRT_2).abs() < 1e-10,
            "Newton's method should converge to sqrt(2), got {sqrt2}"
        );
    }

    #[test]
    fn test_while_loop_zero_iterations() {
        // Condition false initially → returns init unchanged
        let init = Value::scalar_f64(42.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(100.0);
        // carry > 100 is false for carry=42
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }

    #[test]
    fn test_while_loop_single_iteration() {
        // Condition true once, then false
        // carry=5, while carry < 6: carry += 1 → 6, then 6 < 6 is false
        let init = Value::scalar_f64(5.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(6.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 6.0);
    }

    #[test]
    fn test_while_loop_max_iterations() {
        // Loop that would never terminate, but max_iter caps it
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(0.0); // Adding zero never changes carry
        let threshold = Value::scalar_f64(10.0);
        let result = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params_max("add", "lt", 5),
        );
        match result {
            Err(super::EvalError::MaxIterationsExceeded { max_iterations, .. }) => {
                assert_eq!(max_iterations, 5);
            }
            other => panic!("expected MaxIterationsExceeded, got {other:?}"),
        }
    }

    #[test]
    fn test_while_loop_functional_tuple_carry() {
        // Carry state is (i, accumulator): count from 0 to 5, accumulating squares
        let init = vec![Value::scalar_f64(0.0), Value::scalar_f64(0.0)];
        let result = super::eval_while_loop_functional(
            init,
            100,
            |carry| {
                let i = carry[0].as_f64_scalar().unwrap();
                Ok(i < 5.0)
            },
            |carry| {
                let i = carry[0].as_f64_scalar().unwrap();
                let acc = carry[1].as_f64_scalar().unwrap();
                Ok(vec![
                    Value::scalar_f64(i + 1.0),
                    Value::scalar_f64(acc + i * i),
                ])
            },
        )
        .unwrap();
        let final_i = result[0].as_f64_scalar().unwrap();
        let final_acc = result[1].as_f64_scalar().unwrap();
        assert_eq!(final_i, 5.0);
        // 0² + 1² + 2² + 3² + 4² = 30
        assert_eq!(final_acc, 30.0);
    }

    #[test]
    fn test_while_loop_functional_shape_mismatch() {
        // Body changes carry shape → error
        let init = vec![Value::scalar_f64(0.0)];
        let result = super::eval_while_loop_functional(
            init,
            100,
            |_carry| Ok(true),
            |_carry| {
                // Return a vector instead of scalar
                Ok(vec![Value::vector_f64(&[1.0, 2.0]).unwrap()])
            },
        );
        match result {
            Err(super::EvalError::ShapeChanged { .. }) => {}
            other => panic!("expected ShapeChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_while_loop_div_body_op() {
        // while carry > 1: carry /= 2 → 16, 8, 4, 2, 1 → stops at 1
        let init = Value::scalar_f64(16.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(1.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("div", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 1.0);
    }

    #[test]
    fn test_while_loop_ne_cond() {
        // while carry != 10: carry += 2 → 0, 2, 4, 6, 8, 10 → stops at 10
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(10.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "ne"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    // ── Bitwise tests ───────────────────────────────────────────────

    #[test]
    fn bitwise_and_scalars() {
        let a = Value::scalar_i64(0b1100);
        let b = Value::scalar_i64(0b1010);
        let out = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0b1000);
    }

    #[test]
    fn bitwise_or_scalars() {
        let a = Value::scalar_i64(0b1100);
        let b = Value::scalar_i64(0b1010);
        let out = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0b1110);
    }

    #[test]
    fn bitwise_xor_scalars() {
        let a = Value::scalar_i64(0b1100);
        let b = Value::scalar_i64(0b1010);
        let out = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0b0110);
    }

    #[test]
    fn bitwise_not_scalar() {
        let a = Value::scalar_i64(0);
        let out = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), -1); // !0 = all ones = -1 in two's complement
    }

    #[test]
    fn shift_left_scalar() {
        let a = Value::scalar_i64(1);
        let b = Value::scalar_i64(4);
        let out = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 16);
    }

    #[test]
    fn shift_right_scalar() {
        let a = Value::scalar_i64(16);
        let b = Value::scalar_i64(2);
        let out = eval_primitive(Primitive::ShiftRight, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 4);
    }

    #[test]
    fn bitwise_type_error_f64() {
        let a = Value::scalar_f64(1.0);
        let b = Value::scalar_f64(2.0);
        let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params());
        assert!(result.is_err());
    }

    // ── PopulationCount / CountLeadingZeros tests ─────────────────────

    #[test]
    fn population_count_scalar() {
        let a = Value::scalar_i64(0b1010_1100);
        let out = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 4);
    }

    #[test]
    fn population_count_zero() {
        let a = Value::scalar_i64(0);
        let out = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0);
    }

    #[test]
    fn population_count_all_ones() {
        let a = Value::scalar_i64(-1); // all bits set
        let out = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 64);
    }

    #[test]
    fn count_leading_zeros_scalar() {
        let a = Value::scalar_i64(1);
        let out = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 63);
    }

    #[test]
    fn count_leading_zeros_zero() {
        let a = Value::scalar_i64(0);
        let out = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 64);
    }

    #[test]
    fn count_leading_zeros_negative() {
        let a = Value::scalar_i64(-1); // all ones, no leading zeros
        let out = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0);
    }

    #[test]
    fn population_count_type_error() {
        let a = Value::scalar_f64(1.0);
        let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params());
        assert!(result.is_err());
    }

    // ── ReduceWindow tests ──────────────────────────────────────────

    fn rw_params(reduce_op: &str, window: &str, strides: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("reduce_op".to_owned(), reduce_op.to_owned());
        p.insert("window_dimensions".to_owned(), window.to_owned());
        p.insert("window_strides".to_owned(), strides.to_owned());
        p
    }

    #[test]
    fn reduce_window_sum_1d() {
        // [1, 2, 3, 4, 5], window=3, stride=1, valid => [6, 9, 12]
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "3", "1"),
        )
        .unwrap();
        if let Value::Tensor(t) = out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![6.0, 9.0, 12.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn reduce_window_max_1d() {
        // [1, 3, 2, 5, 4], window=2, stride=1, valid => [3, 3, 5, 5]
        let input = Value::vector_f64(&[1.0, 3.0, 2.0, 5.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("max", "2", "1"),
        )
        .unwrap();
        if let Value::Tensor(t) = out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 3.0, 5.0, 5.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn reduce_window_max_1d_stride2() {
        // [1, 3, 2, 5, 4, 6], window=2, stride=2, valid => [3, 5, 6]
        let input = Value::vector_f64(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("max", "2", "2"),
        )
        .unwrap();
        if let Value::Tensor(t) = out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 5.0, 6.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn reduce_window_sum_2d() {
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        // window=2x2, stride=1x1 => [[12, 16], [24, 28]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 3] },
                (1..=9).map(|v| Literal::from_f64(v as f64)).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2,2", "1,1"),
        )
        .unwrap();
        if let Value::Tensor(t) = out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![12.0, 16.0, 24.0, 28.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn reduce_window_min_1d() {
        // [5, 3, 7, 1, 4], window=3, stride=1 => [3, 1, 1]
        let input = Value::vector_f64(&[5.0, 3.0, 7.0, 1.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("min", "3", "1"),
        )
        .unwrap();
        if let Value::Tensor(t) = out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 1.0, 1.0]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn reduce_window_scalar_passthrough() {
        let input = Value::scalar_f64(42.0);
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "1", "1"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::eval_primitive;
    use fj_core::{Primitive, Value};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    proptest! {
        #[test]
        fn prop_add_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_mul_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Mul,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Mul,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_max_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Max,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Max,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_min_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Min,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Min,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_neg_involution(x in -1000i64..1000) {
            let neg1 = eval_primitive(
                Primitive::Neg,
                &[Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            let neg2 = eval_primitive(
                Primitive::Neg,
                &[neg1],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(neg2, Value::scalar_i64(x));
        }

        #[test]
        fn prop_abs_non_negative(x in -1000i64..1000) {
            let out = eval_primitive(
                Primitive::Abs,
                &[Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            if let Value::Scalar(fj_core::Literal::I64(v)) = out {
                prop_assert!(v >= 0, "abs({x}) = {v} should be non-negative");
            }
        }

        #[test]
        fn prop_reshape_roundtrip(a in -100i64..100, b in -100i64..100, c in -100i64..100) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();

            let mut to_3x1 = BTreeMap::new();
            to_3x1.insert("new_shape".into(), "3,1".into());
            let reshaped = eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &to_3x1).unwrap();

            let mut to_3 = BTreeMap::new();
            to_3.insert("new_shape".into(), "3".into());
            let restored = eval_primitive(Primitive::Reshape, &[reshaped], &to_3).unwrap();

            prop_assert_eq!(restored, input);
        }

        #[test]
        fn prop_reduce_sum_matches_manual(
            a in -100i64..100,
            b in -100i64..100,
            c in -100i64..100
        ) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();
            let out = eval_primitive(
                Primitive::ReduceSum,
                &[input],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_i64(a + b + c));
        }

        #[test]
        fn prop_reduce_prod_matches_manual(
            a in -10i64..10,
            b in -10i64..10,
            c in -10i64..10
        ) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();
            let out = eval_primitive(
                Primitive::ReduceProd,
                &[input],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_i64(a * b * c));
        }

        #[test]
        fn prop_eq_reflexive(x in -1000i64..1000) {
            let out = eval_primitive(
                Primitive::Eq,
                &[Value::scalar_i64(x), Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_bool(true));
        }

        #[test]
        fn prop_add_sub_inverse(a in -1000i64..1000, b in -1000i64..1000) {
            let sum = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let result = eval_primitive(
                Primitive::Sub,
                &[sum, Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(result, Value::scalar_i64(a));
        }
    }
}
