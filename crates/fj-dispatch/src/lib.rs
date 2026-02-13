#![forbid(unsafe_code)]

use fj_cache::{CacheKeyError, CacheKeyInput, build_cache_key};
use fj_core::{
    CompatibilityMode, DType, Jaxpr, TensorValue, TraceTransformLedger, Transform,
    TransformCompositionError, Value, verify_transform_composition,
};
use fj_interpreters::{InterpreterError, eval_jaxpr};
use fj_ledger::{DecisionRecord, EvidenceLedger, EvidenceSignal, LedgerEntry, LossMatrix};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRequest {
    pub mode: CompatibilityMode,
    pub ledger: TraceTransformLedger,
    pub args: Vec<Value>,
    pub backend: String,
    pub compile_options: BTreeMap<String, String>,
    pub custom_hook: Option<String>,
    pub unknown_incompatible_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchResponse {
    pub outputs: Vec<Value>,
    pub cache_key: String,
    pub evidence_ledger: EvidenceLedger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformExecutionError {
    EmptyArgumentList { transform: Transform },
    NonScalarGradientInput,
    NonScalarGradientOutput,
    VmapRequiresRankOneLeadingArgument,
    VmapMismatchedLeadingDimension { expected: usize, actual: usize },
    VmapNonScalarOutput,
    VmapInconsistentOutputArity { expected: usize, actual: usize },
    EmptyVmapOutput,
    TensorBuild(String),
}

impl std::fmt::Display for TransformExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyArgumentList { transform } => {
                write!(f, "{} requires at least one argument", transform.as_str())
            }
            Self::NonScalarGradientInput => {
                write!(f, "grad currently requires scalar first input")
            }
            Self::NonScalarGradientOutput => {
                write!(f, "grad currently requires scalar first output")
            }
            Self::VmapRequiresRankOneLeadingArgument => {
                write!(f, "vmap currently requires rank-1 first argument")
            }
            Self::VmapMismatchedLeadingDimension { expected, actual } => {
                write!(
                    f,
                    "vmap leading-dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::VmapNonScalarOutput => {
                write!(
                    f,
                    "vmap inner function must currently return scalar outputs"
                )
            }
            Self::VmapInconsistentOutputArity { expected, actual } => {
                write!(
                    f,
                    "vmap inner output arity mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::EmptyVmapOutput => {
                write!(f, "vmap received no mapped elements")
            }
            Self::TensorBuild(detail) => write!(f, "tensor build error: {detail}"),
        }
    }
}

impl std::error::Error for TransformExecutionError {}

#[derive(Debug)]
pub enum DispatchError {
    Cache(CacheKeyError),
    Interpreter(InterpreterError),
    TransformInvariant(TransformCompositionError),
    TransformExecution(TransformExecutionError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cache(err) => write!(f, "cache key error: {err}"),
            Self::Interpreter(err) => write!(f, "interpreter error: {err}"),
            Self::TransformInvariant(err) => write!(f, "transform invariant error: {err}"),
            Self::TransformExecution(err) => write!(f, "transform execution error: {err}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<CacheKeyError> for DispatchError {
    fn from(value: CacheKeyError) -> Self {
        Self::Cache(value)
    }
}

impl From<InterpreterError> for DispatchError {
    fn from(value: InterpreterError) -> Self {
        Self::Interpreter(value)
    }
}

impl From<TransformCompositionError> for DispatchError {
    fn from(value: TransformCompositionError) -> Self {
        Self::TransformInvariant(value)
    }
}

impl From<TransformExecutionError> for DispatchError {
    fn from(value: TransformExecutionError) -> Self {
        Self::TransformExecution(value)
    }
}

pub fn dispatch(request: DispatchRequest) -> Result<DispatchResponse, DispatchError> {
    let composition_proof = verify_transform_composition(&request.ledger)?;

    let cache_key = build_cache_key(&CacheKeyInput {
        mode: request.mode,
        backend: request.backend,
        jaxpr: request.ledger.root_jaxpr.clone(),
        transform_stack: request.ledger.transform_stack.clone(),
        compile_options: request.compile_options,
        custom_hook: request.custom_hook,
        unknown_incompatible_features: request.unknown_incompatible_features,
    })?;

    let outputs = execute_with_transforms(
        &request.ledger.root_jaxpr,
        &request.ledger.transform_stack,
        &request.args,
    )?;

    let mut evidence_ledger = EvidenceLedger::new();
    let posterior_abandoned = heuristic_posterior_abandoned(&request.ledger);
    let matrix = LossMatrix::default();
    let record = DecisionRecord::from_posterior(request.mode, posterior_abandoned, &matrix);

    evidence_ledger.append(LedgerEntry {
        decision_id: cache_key.as_string(),
        record,
        signals: vec![
            EvidenceSignal {
                signal_name: "eqn_count".to_owned(),
                log_likelihood_delta: (request.ledger.root_jaxpr.equations.len() as f64 + 1.0).ln(),
                detail: format!("eqn_count={}", request.ledger.root_jaxpr.equations.len()),
            },
            EvidenceSignal {
                signal_name: "transform_depth".to_owned(),
                log_likelihood_delta: request.ledger.transform_stack.len() as f64 * 0.1,
                detail: format!("transform_depth={}", request.ledger.transform_stack.len()),
            },
            EvidenceSignal {
                signal_name: "transform_stack_hash".to_owned(),
                log_likelihood_delta: (composition_proof.transform_count as f64 + 1.0).ln(),
                detail: composition_proof.stack_hash_hex,
            },
        ],
    });

    Ok(DispatchResponse {
        outputs,
        cache_key: cache_key.as_string(),
        evidence_ledger,
    })
}

fn execute_with_transforms(
    root_jaxpr: &Jaxpr,
    transforms: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError> {
    let Some((head, tail)) = transforms.split_first() else {
        return eval_jaxpr(root_jaxpr, args).map_err(DispatchError::from);
    };

    match head {
        Transform::Jit => execute_with_transforms(root_jaxpr, tail, args),
        Transform::Grad => execute_grad(root_jaxpr, tail, args),
        Transform::Vmap => execute_vmap(root_jaxpr, tail, args),
    }
}

fn execute_grad(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Grad,
        }
        .into());
    }

    let input_value = args[0]
        .as_f64_scalar()
        .ok_or(TransformExecutionError::NonScalarGradientInput)?;

    let epsilon = 1e-6_f64;
    let mut plus_args = args.to_vec();
    let mut minus_args = args.to_vec();
    plus_args[0] = Value::scalar_f64(input_value + epsilon);
    minus_args[0] = Value::scalar_f64(input_value - epsilon);

    let plus_out = execute_with_transforms(root_jaxpr, tail, &plus_args)?;
    let minus_out = execute_with_transforms(root_jaxpr, tail, &minus_args)?;

    let plus_value = plus_out
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(TransformExecutionError::NonScalarGradientOutput)?;
    let minus_value = minus_out
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(TransformExecutionError::NonScalarGradientOutput)?;

    let derivative = (plus_value - minus_value) / (2.0 * epsilon);
    Ok(vec![Value::scalar_f64(derivative)])
}

fn execute_vmap(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Vmap,
        }
        .into());
    }

    let lead_tensor = args[0]
        .as_tensor()
        .ok_or(TransformExecutionError::VmapRequiresRankOneLeadingArgument)?;
    if lead_tensor.rank() != 1 {
        return Err(TransformExecutionError::VmapRequiresRankOneLeadingArgument.into());
    }

    let lead_len = lead_tensor.len();
    if lead_len == 0 {
        return Err(TransformExecutionError::EmptyVmapOutput.into());
    }

    let mut per_output_scalars: Vec<Vec<fj_core::Literal>> = Vec::new();

    for index in 0..lead_len {
        let mut mapped_args = Vec::with_capacity(args.len());
        mapped_args.push(Value::Scalar(lead_tensor.elements[index]));

        for arg in &args[1..] {
            match arg {
                Value::Scalar(lit) => mapped_args.push(Value::Scalar(*lit)),
                Value::Tensor(tensor) => {
                    if tensor.rank() != 1 {
                        return Err(
                            TransformExecutionError::VmapRequiresRankOneLeadingArgument.into()
                        );
                    }
                    if tensor.len() != lead_len {
                        return Err(TransformExecutionError::VmapMismatchedLeadingDimension {
                            expected: lead_len,
                            actual: tensor.len(),
                        }
                        .into());
                    }
                    mapped_args.push(Value::Scalar(tensor.elements[index]));
                }
            }
        }

        let mapped_output = execute_with_transforms(root_jaxpr, tail, &mapped_args)?;
        if index == 0 {
            per_output_scalars = vec![Vec::with_capacity(lead_len); mapped_output.len()];
        } else if mapped_output.len() != per_output_scalars.len() {
            return Err(TransformExecutionError::VmapInconsistentOutputArity {
                expected: per_output_scalars.len(),
                actual: mapped_output.len(),
            }
            .into());
        }

        for (output_idx, value) in mapped_output.iter().enumerate() {
            let scalar = value
                .as_scalar_literal()
                .ok_or(TransformExecutionError::VmapNonScalarOutput)?;
            per_output_scalars[output_idx].push(scalar);
        }
    }

    let mut outputs = Vec::with_capacity(per_output_scalars.len());
    for scalars in per_output_scalars {
        let dtype = if scalars
            .iter()
            .all(|literal| matches!(literal, fj_core::Literal::I64(_)))
        {
            DType::I64
        } else {
            DType::F64
        };

        let tensor = TensorValue::new(dtype, fj_core::Shape::vector(lead_len as u32), scalars)
            .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
        outputs.push(Value::Tensor(tensor));
    }

    Ok(outputs)
}

fn heuristic_posterior_abandoned(ledger: &TraceTransformLedger) -> f64 {
    let eqn_factor = ledger.root_jaxpr.equations.len() as f64;
    let depth_factor = ledger.transform_stack.len() as f64;
    let score = (eqn_factor + 2.0 * depth_factor) / (eqn_factor + 2.0 * depth_factor + 20.0);
    score.clamp(0.05, 0.95)
}

#[cfg(test)]
mod tests {
    use super::{DispatchError, DispatchRequest, dispatch};
    use fj_core::{
        CompatibilityMode, ProgramSpec, TraceTransformLedger, Transform, Value, build_program,
    };
    use std::collections::BTreeMap;

    fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
        let mut ledger = TraceTransformLedger::new(build_program(program));
        for (idx, transform) in transforms.iter().enumerate() {
            ledger.push_transform(
                *transform,
                format!("evidence-{}-{}", transform.as_str(), idx),
            );
        }
        ledger
    }

    #[test]
    fn dispatch_jit_add_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
        assert!(response.cache_key.starts_with("fjx-"));
        assert_eq!(response.evidence_ledger.len(), 1);
    }

    #[test]
    fn dispatch_grad_square_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad should succeed");

        let derivative = response.outputs[0]
            .as_f64_scalar()
            .expect("grad output should be scalar f64");
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_vmap_add_one_vector() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4]);
    }

    #[test]
    fn transform_order_is_explicit() {
        let good = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Vmap, Transform::Grad]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(grad(f)) should be supported");

        let good_out = good.outputs[0]
            .as_tensor()
            .expect("output should be tensor")
            .to_f64_vec()
            .expect("output should be numeric");
        assert_eq!(good_out.len(), 3);
        assert!((good_out[0] - 2.0).abs() < 1e-3);
        assert!((good_out[1] - 4.0).abs() < 1e-3);
        assert!((good_out[2] - 6.0).abs() < 1e-3);

        let bad = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Vmap]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("grad(vmap(f)) should fail with current constraints");

        assert!(matches!(
            bad,
            DispatchError::TransformExecution(
                super::TransformExecutionError::NonScalarGradientInput
            )
        ));
    }
}
