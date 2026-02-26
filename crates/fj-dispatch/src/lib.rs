#![forbid(unsafe_code)]

pub mod batching;

use fj_cache::{CacheKeyError, CacheKeyInputRef, build_cache_key_ref};
use fj_core::{
    CompatibilityMode, Jaxpr, TensorValue, TraceTransformLedger, Transform,
    TransformCompositionError, Value, verify_transform_composition,
};
use fj_interpreters::InterpreterError;
use fj_ledger::{
    ConformalPredictor, DecisionRecord, EvidenceLedger, EvidenceSignal, LedgerEntry, LossMatrix,
};
use fj_runtime::backend::{Backend, BackendError, BackendRegistry};
use fj_runtime::device::{DeviceId, DevicePlacement};
use std::collections::BTreeMap;

// ── Effect Token System ────────────────────────────────────────────

/// Per-effect runtime token for ordered side-effect tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectToken {
    pub effect_name: String,
    pub sequence_number: u64,
}

/// Context for threading effect tokens through a dispatch execution.
///
/// V1 scope: tracking only — records which effects were observed and in what
/// order. No execution ordering is enforced (effects modeled via evidence
/// ledger signals rather than runtime token threading).
///
/// Uses a Vec instead of BTreeMap since transform stacks are small (typically
/// 1-3 elements) and insertion-order tracking eliminates the need for sorting.
#[derive(Debug, Clone)]
pub struct EffectContext {
    tokens: Vec<EffectToken>,
    next_sequence: u64,
}

impl EffectContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            next_sequence: 0,
        }
    }

    /// Record observation of a named effect. Returns the token with its
    /// sequence number in the observation order. If the same effect name
    /// was already observed, overwrites the previous entry (BTreeMap parity).
    pub fn thread_token(&mut self, effect_name: &str) -> EffectToken {
        let name = effect_name.to_owned();
        let token = EffectToken {
            effect_name: name,
            sequence_number: self.next_sequence,
        };
        self.next_sequence += 1;
        // Overwrite existing entry with same name (preserves BTreeMap semantics).
        if let Some(pos) = self
            .tokens
            .iter()
            .position(|t| t.effect_name == token.effect_name)
        {
            self.tokens.remove(pos);
        }
        self.tokens.push(token.clone());
        token
    }

    /// Finalize and return all observed effect tokens in sequence order.
    /// Tokens are already in insertion order, so no sorting needed.
    #[must_use]
    pub fn finalize(self) -> Vec<EffectToken> {
        self.tokens
    }

    /// Number of distinct effects observed.
    #[must_use]
    pub fn effect_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Default for EffectContext {
    fn default() -> Self {
        Self::new()
    }
}

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
                write!(f, "vmap currently requires first argument with rank >= 1")
            }
            Self::VmapMismatchedLeadingDimension { expected, actual } => {
                write!(
                    f,
                    "vmap leading-dimension mismatch: expected {}, got {}",
                    expected, actual
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
    BackendExecution(BackendError),
    TransformInvariant(TransformCompositionError),
    TransformExecution(TransformExecutionError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cache(err) => write!(f, "cache key error: {err}"),
            Self::Interpreter(err) => write!(f, "interpreter error: {err}"),
            Self::BackendExecution(err) => write!(f, "backend execution error: {err}"),
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

impl From<BackendError> for DispatchError {
    fn from(value: BackendError) -> Self {
        Self::BackendExecution(value)
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

    let cache_key = build_cache_key_ref(&CacheKeyInputRef {
        mode: request.mode,
        backend: &request.backend,
        jaxpr: &request.ledger.root_jaxpr,
        transform_stack: &request.ledger.transform_stack,
        compile_options: &request.compile_options,
        custom_hook: request.custom_hook.as_deref(),
        unknown_incompatible_features: &request.unknown_incompatible_features,
    })?;

    // Thread effect context through transform execution
    let mut effect_ctx = EffectContext::new();
    for transform in &request.ledger.transform_stack {
        effect_ctx.thread_token(transform.as_str());
    }

    let backend_registry = BackendRegistry::new(vec![Box::new(fj_backend_cpu::CpuBackend::new())]);
    let requested_backend = (!request.backend.is_empty()).then_some(request.backend.as_str());
    let (backend, device, _fell_back) =
        backend_registry.resolve_with_fallback(&DevicePlacement::Default, requested_backend)?;
    let outputs = execute_with_transforms(
        &request.ledger.root_jaxpr,
        &request.ledger.transform_stack,
        &request.args,
        backend,
        device,
    )?;

    let effect_tokens = effect_ctx.finalize();
    let effect_count = effect_tokens.len();

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
            EvidenceSignal {
                signal_name: "effect_token_count".to_owned(),
                log_likelihood_delta: (effect_count as f64 + 1.0).ln(),
                detail: format!("effect_tokens={effect_count}"),
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
    backend: &dyn Backend,
    device: DeviceId,
) -> Result<Vec<Value>, DispatchError> {
    // Skip leading Jit transforms (no-op pass-through) to find the first
    // non-Jit transform, avoiding recursive stack frames for Jit chains.
    let non_jit_start = transforms
        .iter()
        .position(|t| *t != Transform::Jit)
        .unwrap_or(transforms.len());

    let remaining = &transforms[non_jit_start..];
    let Some((head, tail)) = remaining.split_first() else {
        return backend
            .execute(root_jaxpr, args, device)
            .map_err(DispatchError::from);
    };

    match head {
        Transform::Jit => unreachable!("Jit transforms were skipped above"),
        Transform::Grad => execute_grad(root_jaxpr, tail, args, backend, device),
        Transform::Vmap => execute_vmap(root_jaxpr, tail, args, backend, device),
    }
}

fn execute_grad(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Grad,
        }
        .into());
    }

    // If there are remaining transforms in the tail, fall back to finite-diff
    // (symbolic AD only applies to the innermost evaluation).
    // For finite-diff, we still require scalar first argument.
    if !tail.is_empty() {
        args[0]
            .as_f64_scalar()
            .ok_or(TransformExecutionError::NonScalarGradientInput)?;
        return execute_grad_finite_diff(root_jaxpr, tail, args, backend, device);
    }

    // Tensor-aware AD: grad_jaxpr returns Value gradients for all inputs
    let grads = fj_ad::grad_jaxpr(root_jaxpr, args).map_err(|e| match e {
        fj_ad::AdError::NonScalarGradientOutput => TransformExecutionError::NonScalarGradientOutput,
        other => TransformExecutionError::TensorBuild(format!("AD error: {other}")),
    })?;
    // Return gradient of first input (matches JAX's default grad behavior)
    Ok(vec![
        grads.into_iter().next().unwrap_or(Value::scalar_f64(0.0)),
    ])
}

fn execute_grad_finite_diff(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
) -> Result<Vec<Value>, DispatchError> {
    let input_value = args[0]
        .as_f64_scalar()
        .ok_or(TransformExecutionError::NonScalarGradientInput)?;

    let epsilon = 1e-6_f64;
    let mut plus_args = args.to_vec();
    let mut minus_args = args.to_vec();
    plus_args[0] = Value::scalar_f64(input_value + epsilon);
    minus_args[0] = Value::scalar_f64(input_value - epsilon);

    let plus_out = execute_with_transforms(root_jaxpr, tail, &plus_args, backend, device)?;
    let minus_out = execute_with_transforms(root_jaxpr, tail, &minus_args, backend, device)?;

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
    backend: &dyn Backend,
    device: DeviceId,
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Vmap,
        }
        .into());
    }

    // Validate inputs: at least the first arg must be a tensor with rank >= 1
    let lead_tensor = args[0]
        .as_tensor()
        .ok_or(TransformExecutionError::VmapRequiresRankOneLeadingArgument)?;
    if lead_tensor.rank() == 0 {
        return Err(TransformExecutionError::VmapRequiresRankOneLeadingArgument.into());
    }

    let lead_len = lead_tensor
        .leading_dim()
        .ok_or(TransformExecutionError::VmapRequiresRankOneLeadingArgument)?
        as usize;
    if lead_len == 0 {
        return Err(TransformExecutionError::EmptyVmapOutput.into());
    }

    // Validate all tensor args have matching leading dimension
    for arg in &args[1..] {
        if let Value::Tensor(tensor) = arg {
            if tensor.rank() == 0 {
                return Err(TransformExecutionError::VmapRequiresRankOneLeadingArgument.into());
            }
            let arg_lead = tensor
                .leading_dim()
                .ok_or(TransformExecutionError::VmapRequiresRankOneLeadingArgument)?
                as usize;
            if arg_lead != lead_len {
                return Err(TransformExecutionError::VmapMismatchedLeadingDimension {
                    expected: lead_len,
                    actual: arg_lead,
                }
                .into());
            }
        }
    }

    // Use BatchTrace interpreter when no tail transforms exist.
    // This gives O(1) vectorized execution for the innermost vmap.
    if tail.is_empty() {
        return execute_vmap_batch_trace(root_jaxpr, args);
    }

    // When there are tail transforms (e.g., vmap(grad(f))), fall back to
    // loop-and-stack since BatchTrace operates at the Jaxpr primitive level
    // and cannot handle composed transforms within the trace.
    execute_vmap_loop_and_stack(root_jaxpr, tail, args, backend, device, lead_len)
}

/// BatchTrace-based vmap execution: O(1) vectorized dispatch via per-primitive
/// batching rules. Each arg becomes a BatchTracer with batch_dim=0 (tensors)
/// or None (scalars), and the Jaxpr is evaluated equation-by-equation.
fn execute_vmap_batch_trace(
    root_jaxpr: &Jaxpr,
    args: &[Value],
) -> Result<Vec<Value>, DispatchError> {
    use batching::{BatchTracer, batch_eval_jaxpr};

    let batch_inputs: Vec<BatchTracer> = args
        .iter()
        .map(|arg| match arg {
            Value::Scalar(_) => BatchTracer::unbatched(arg.clone()),
            Value::Tensor(_) => BatchTracer::batched(arg.clone(), 0),
        })
        .collect();

    let results = batch_eval_jaxpr(root_jaxpr, &batch_inputs)
        .map_err(|e| TransformExecutionError::TensorBuild(format!("BatchTrace error: {e}")))?;

    // Extract output values, ensuring batch dim is at position 0
    let mut outputs = Vec::with_capacity(results.len());
    for tracer in results {
        match tracer.batch_dim {
            Some(0) => outputs.push(tracer.value),
            Some(bd) => {
                // Move batch dim to front for consistent output
                let moved = batching::move_batch_dim_to_front(&tracer.value, bd)
                    .map_err(|e| TransformExecutionError::TensorBuild(e.to_string()))?;
                outputs.push(moved);
            }
            None => {
                // Unbatched output — need to broadcast to match batch dimension
                // This happens when the function output doesn't depend on the input
                outputs.push(tracer.value);
            }
        }
    }

    Ok(outputs)
}

/// Loop-and-stack vmap fallback for composed transforms (e.g., vmap(grad(f))).
fn execute_vmap_loop_and_stack(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    lead_len: usize,
) -> Result<Vec<Value>, DispatchError> {
    let lead_tensor = args[0].as_tensor().unwrap();
    let mut per_output_values: Vec<Vec<Value>> = Vec::new();

    for index in 0..lead_len {
        let mut mapped_args = Vec::with_capacity(args.len());
        mapped_args.push(
            lead_tensor
                .slice_axis0(index)
                .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?,
        );

        for arg in &args[1..] {
            match arg {
                Value::Scalar(lit) => mapped_args.push(Value::Scalar(*lit)),
                Value::Tensor(tensor) => {
                    mapped_args.push(
                        tensor
                            .slice_axis0(index)
                            .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?,
                    );
                }
            }
        }

        let mapped_output =
            execute_with_transforms(root_jaxpr, tail, &mapped_args, backend, device)?;
        if index == 0 {
            per_output_values = vec![Vec::with_capacity(lead_len); mapped_output.len()];
        } else if mapped_output.len() != per_output_values.len() {
            return Err(TransformExecutionError::VmapInconsistentOutputArity {
                expected: per_output_values.len(),
                actual: mapped_output.len(),
            }
            .into());
        }

        for (output_idx, value) in mapped_output.iter().enumerate() {
            per_output_values[output_idx].push(value.clone());
        }
    }

    let mut outputs = Vec::with_capacity(per_output_values.len());
    for values in per_output_values {
        let tensor = TensorValue::stack_axis0(&values)
            .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
        outputs.push(Value::Tensor(tensor));
    }

    Ok(outputs)
}

#[inline]
fn heuristic_posterior_abandoned(ledger: &TraceTransformLedger) -> f64 {
    let eqn_factor = ledger.root_jaxpr.equations.len() as f64;
    let depth_factor = ledger.transform_stack.len() as f64;
    let score = (eqn_factor + 2.0 * depth_factor) / (eqn_factor + 2.0 * depth_factor + 20.0);
    score.clamp(0.05, 0.95)
}

/// Compute posterior with conformal calibration when available.
#[must_use]
pub fn calibrated_posterior_abandoned(
    ledger: &TraceTransformLedger,
    conformal: Option<&ConformalPredictor>,
) -> f64 {
    let heuristic = heuristic_posterior_abandoned(ledger);
    match conformal {
        Some(cp) if cp.is_calibrated() => {
            let estimate = cp.calibrated_posterior(heuristic);
            estimate.point
        }
        _ => heuristic,
    }
}

#[cfg(test)]
mod tests {
    use super::{DispatchError, DispatchRequest, dispatch};
    use fj_core::{
        CompatibilityMode, DType, ProgramSpec, Shape, TensorValue, TraceTransformLedger, Transform,
        Value, build_program,
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
    fn dispatch_grad_of_grad_square_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad-of-grad should succeed");

        let second_derivative = response.outputs[0]
            .as_f64_scalar()
            .expect("second derivative output should be scalar f64");
        assert!((second_derivative - 2.0).abs() < 1e-3);
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
    fn dispatch_vmap_add_one_rank2_tensor() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                ],
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap rank2 should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![2, 2] });
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4, 5]);
    }

    #[test]
    fn dispatch_vmap_of_vmap_add_one_rank2_tensor() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                    fj_core::Literal::I64(5),
                    fj_core::Literal::I64(6),
                ],
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch nested vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("nested vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![2, 3] });
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4, 5, 6, 7]);
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

    #[test]
    fn strict_mode_rejects_unknown_features_fail_closed() {
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["future.backend.protocol.v2".to_owned()],
        })
        .expect_err("strict mode must reject unknown incompatible features");

        match err {
            DispatchError::Cache(fj_cache::CacheKeyError::UnknownIncompatibleFeatures {
                features,
            }) => {
                assert_eq!(features, vec!["future.backend.protocol.v2".to_owned()]);
            }
            other => {
                panic!("expected fail-closed cache-key rejection, got: {other:?}");
            }
        }
    }

    #[test]
    fn hardened_mode_allowlists_unknown_features_for_auditable_progress() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["future.backend.protocol.v2".to_owned()],
        })
        .expect("hardened mode should permit allowlisted unknown features");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
        assert!(response.cache_key.starts_with("fjx-"));
        assert_eq!(response.evidence_ledger.len(), 1);
    }

    #[test]
    fn test_dispatch_test_log_schema_contract() {
        let fixture_id = fj_test_utils::fixture_id_from_json(&("dispatch", "transform-order"))
            .expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_dispatch_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Effect token tests ─────────────────────────────────────

    #[test]
    fn effect_context_tracks_transform_tokens() {
        use super::EffectContext;
        let mut ctx = EffectContext::new();
        let t1 = ctx.thread_token("jit");
        let t2 = ctx.thread_token("grad");
        let t3 = ctx.thread_token("vmap");

        assert_eq!(t1.sequence_number, 0);
        assert_eq!(t2.sequence_number, 1);
        assert_eq!(t3.sequence_number, 2);
        assert_eq!(ctx.effect_count(), 3);

        let tokens = ctx.finalize();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].effect_name, "jit");
        assert_eq!(tokens[1].effect_name, "grad");
        assert_eq!(tokens[2].effect_name, "vmap");
    }

    #[test]
    fn dispatch_includes_effect_token_signal() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        // Evidence ledger should have 1 entry with 4 signals (including effect_token_count)
        assert_eq!(response.evidence_ledger.len(), 1);
        let entry = &response.evidence_ledger.entries()[0];
        assert_eq!(entry.signals.len(), 4);

        let effect_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_count")
            .expect("should have effect_token_count signal");
        assert_eq!(effect_signal.detail, "effect_tokens=2");
    }

    #[test]
    fn dispatch_cache_hit_miss_determinism() {
        // Two identical requests should produce identical cache keys
        let make_request = || DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };

        let r1 = dispatch(make_request()).expect("dispatch 1");
        let r2 = dispatch(make_request()).expect("dispatch 2");
        assert_eq!(r1.cache_key, r2.cache_key, "same request = same cache key");
        assert_eq!(r1.outputs, r2.outputs, "same request = same outputs");

        // Different program should produce different cache key
        let r3 = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit]),
            args: vec![Value::scalar_f64(2.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch 3");
        assert_ne!(
            r1.cache_key, r3.cache_key,
            "different program = different key"
        );
    }

    #[test]
    fn dispatch_unknown_backend_falls_back_to_cpu_execution() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "quantum".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("unknown backend should fall back to cpu backend");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
    }

    #[test]
    fn dispatch_backend_name_still_changes_cache_key_under_fallback() {
        let make_request = |backend: &str| DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: backend.to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };

        let cpu = dispatch(make_request("cpu")).expect("cpu dispatch should succeed");
        let fallback = dispatch(make_request("quantum")).expect("fallback dispatch should succeed");

        assert_eq!(cpu.outputs, fallback.outputs);
        assert_ne!(
            cpu.cache_key, fallback.cache_key,
            "requested backend remains part of cache identity even when runtime execution falls back"
        );
    }

    // ── Calibrated posterior tests ────────────────────────────

    #[test]
    fn calibrated_posterior_falls_back_to_heuristic_without_conformal() {
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let result = super::calibrated_posterior_abandoned(&ttl, None);
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        assert!(
            (result - heuristic).abs() < 1e-12,
            "without conformal predictor, should return heuristic: {result} vs {heuristic}"
        );
    }

    #[test]
    fn calibrated_posterior_falls_back_when_uncalibrated() {
        use fj_ledger::ConformalPredictor;
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let cp = ConformalPredictor::new(0.9, 10); // needs 10 scores, has 0
        let result = super::calibrated_posterior_abandoned(&ttl, Some(&cp));
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        assert!(
            (result - heuristic).abs() < 1e-12,
            "uncalibrated predictor should fall back to heuristic"
        );
    }

    #[test]
    fn calibrated_posterior_uses_conformal_when_calibrated() {
        use fj_ledger::ConformalPredictor;
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let mut cp = ConformalPredictor::new(0.9, 5);
        for score in &[0.01, 0.02, 0.03, 0.04, 0.05] {
            cp.observe(*score);
        }
        assert!(cp.is_calibrated());

        let result = super::calibrated_posterior_abandoned(&ttl, Some(&cp));
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        // Result should equal the conformal point estimate (which equals heuristic)
        // because calibrated_posterior returns point = heuristic
        assert!(
            (result - heuristic).abs() < 1e-12,
            "calibrated conformal point estimate should equal heuristic input"
        );
    }

    #[test]
    fn heuristic_posterior_increases_with_transform_depth() {
        let shallow = ledger(ProgramSpec::Add2, &[Transform::Jit]);
        let deep = ledger(
            ProgramSpec::Add2,
            &[Transform::Jit, Transform::Grad, Transform::Vmap],
        );
        let h_shallow = super::heuristic_posterior_abandoned(&shallow);
        let h_deep = super::heuristic_posterior_abandoned(&deep);
        assert!(
            h_deep > h_shallow,
            "deeper transform stack should have higher abandoned posterior: {h_deep} vs {h_shallow}"
        );
    }

    #[test]
    fn heuristic_posterior_is_bounded() {
        let minimal = ledger(ProgramSpec::AddOne, &[]);
        let result = super::heuristic_posterior_abandoned(&minimal);
        assert!(result >= 0.05, "posterior should be >= 0.05, got {result}");
        assert!(result <= 0.95, "posterior should be <= 0.95, got {result}");
    }

    // ══════════════════════════════════════════════════════════════
    // Higher-rank tensor tests (3D+)
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn dispatch_vmap_add_one_rank3_tensor() {
        // 3D tensor [2, 3, 2] — vmap should process each leading-axis slice
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 3, 2],
                },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap rank3 should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap rank3 output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 3, 2]
            }
        );
        let as_i64: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        let expected: Vec<i64> = (2..=13).collect();
        assert_eq!(as_i64, expected);
    }

    #[test]
    fn dispatch_triple_vmap_add_one_rank3_tensor() {
        // 3D tensor with triple vmap (vmap(vmap(vmap(f))))
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 2, 3],
                },
                (0..12).map(fj_core::Literal::I64).collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::AddOne,
                &[Transform::Vmap, Transform::Vmap, Transform::Vmap],
            ),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch triple vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("triple vmap output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 2, 3]
            }
        );
        let as_i64: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        let expected: Vec<i64> = (1..=12).collect();
        assert_eq!(as_i64, expected);
    }

    #[test]
    fn dispatch_triple_vmap_grad_square_rank3_tensor() {
        let input_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                input_vals
                    .iter()
                    .copied()
                    .map(fj_core::Literal::from_f64)
                    .collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[
                    Transform::Vmap,
                    Transform::Vmap,
                    Transform::Vmap,
                    Transform::Grad,
                ],
            ),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch triple vmap(grad(square)) should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("triple vmap grad output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 2, 2]
            }
        );
        let grads = output.to_f64_vec().expect("output should be f64 tensor");
        let expected: Vec<f64> = input_vals.iter().map(|x| 2.0 * x).collect();
        assert_eq!(grads.len(), expected.len());
        for (idx, (actual, expected)) in grads.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "index {idx}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn dispatch_grad_square_rank3_tensor_rejects_non_scalar_input() {
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                (1..=8)
                    .map(|v| fj_core::Literal::from_f64(v as f64))
                    .collect(),
            )
            .expect("3d tensor should build"),
        );
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("grad(square) on rank3 input should fail");

        let err_msg = err.to_string();
        assert!(
            err_msg.contains("scalar"),
            "rank3 grad failure should mention scalar requirement, got: {err_msg}"
        );
    }

    #[test]
    fn dispatch_grad_sin_with_jit() {
        // grad(jit(sin(x))) = cos(x) at x=0 => 1.0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(0.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad(jit(sin)) should succeed");

        let grad_val = response.outputs[0]
            .as_f64_scalar()
            .expect("grad should be scalar f64");
        assert!(
            (grad_val - 1.0).abs() < 1e-10,
            "grad(sin)(0) should be cos(0) = 1.0, got {grad_val}"
        );
    }

    #[test]
    fn dispatch_vmap_sin_f64_vector() {
        // vmap(sin)([0, pi/2, pi]) = [0, 1, ~0]
        use std::f64::consts::PI;
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Vmap]),
            args: vec![Value::vector_f64(&[0.0, PI / 2.0, PI]).unwrap()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap(sin) should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        let vals: Vec<f64> = output.to_f64_vec().expect("should convert to f64");
        assert!(vals[0].abs() < 1e-10, "sin(0) should be 0");
        assert!((vals[1] - 1.0).abs() < 1e-10, "sin(pi/2) should be 1");
        assert!(vals[2].abs() < 1e-10, "sin(pi) should be ~0");
    }

    // ── VMAP tensor output tests (bd-22lm) ───────────────────────

    #[test]
    fn test_vmap_tensor_output_rank1() {
        // vmap(AddOne) over a [3, 4] matrix: inner function receives rank-1 vectors,
        // returns rank-1 vectors, stacked back to [3, 4].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 4] },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with tensor output should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![3, 4] });
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let expected: Vec<i64> = (2..=13).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_vmap_tensor_output_rank2() {
        // vmap(AddOne) over a [2, 3, 4] rank-3 tensor: inner receives rank-2 matrices,
        // returns rank-2 matrices, stacked back to [2, 3, 4].
        let tensor = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 3, 4],
                },
                (0..24).map(fj_core::Literal::I64).collect(),
            )
            .expect("rank-3 tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![tensor],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with rank-2 inner output should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 3, 4]
            }
        );
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let expected: Vec<i64> = (1..=24).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_vmap_multi_output() {
        // vmap(AddOneMulTwo) over [1, 2, 3]: inner returns (x+1, x*2) per element.
        // Output: two tensors, each of shape [3].
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOneMulTwo, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap multi-output should succeed");

        assert_eq!(response.outputs.len(), 2, "should have two outputs");

        let out0 = response.outputs[0]
            .as_tensor()
            .expect("first output should be tensor");
        let out1 = response.outputs[1]
            .as_tensor()
            .expect("second output should be tensor");

        let vals0: Vec<i64> = out0.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        let vals1: Vec<i64> = out1.elements.iter().map(|l| l.as_i64().unwrap()).collect();

        assert_eq!(vals0, vec![2, 3, 4], "x+1 for [1,2,3]");
        assert_eq!(vals1, vec![2, 4, 6], "x*2 for [1,2,3]");
    }

    #[test]
    fn test_vmap_multi_output_loop_and_stack() {
        // Same as above but force the loop-and-stack path by adding a Jit tail transform.
        let input = Value::vector_i64(&[10, 20, 30]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::AddOneMulTwo,
                &[Transform::Vmap, Transform::Jit],
            ),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap multi-output loop-and-stack should succeed");

        assert_eq!(response.outputs.len(), 2, "should have two outputs");

        let vals0: Vec<i64> = response.outputs[0]
            .as_tensor()
            .expect("first output tensor")
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let vals1: Vec<i64> = response.outputs[1]
            .as_tensor()
            .expect("second output tensor")
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();

        assert_eq!(vals0, vec![11, 21, 31], "x+1 for [10,20,30]");
        assert_eq!(vals1, vec![20, 40, 60], "x*2 for [10,20,30]");
    }

    #[test]
    fn test_vmap_output_shape_batch_prepend() {
        // Verify batch dimension is correctly prepended to inner output shape.
        // vmap(AddOne) on [5, 3] matrix: inner returns [3] vectors → stacked to [5, 3].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5, 3] },
                (0..15)
                    .map(|i| fj_core::Literal::from_f64(i as f64))
                    .collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap shape prepend should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        // Output shape should be [5, 3] = [batch_size, ...inner_output_shape]
        assert_eq!(
            output.shape,
            Shape { dims: vec![5, 3] },
            "batch dim (5) prepended to inner output shape (3)"
        );
    }

    #[test]
    fn test_vmap_identity_preserves_shape() {
        // vmap(identity) should return the same tensor as the input.
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 4] },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Identity, &[Transform::Vmap]),
            args: vec![matrix.clone()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(identity) should succeed");

        assert_eq!(response.outputs.len(), 1);
        let output = &response.outputs[0];
        let out_tensor = output.as_tensor().expect("output should be tensor");
        let in_tensor = matrix.as_tensor().unwrap();
        assert_eq!(
            out_tensor.shape, in_tensor.shape,
            "shape should be preserved"
        );
        assert_eq!(
            out_tensor.elements, in_tensor.elements,
            "elements should be identical"
        );
    }

    #[test]
    fn test_vmap_scalar_output_still_works() {
        // Regression: existing scalar-output vmap behavior should be preserved.
        let input = Value::vector_f64(&[1.0, 4.0, 9.0]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(square) with scalar outputs should still work");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be stacked tensor");
        let vals = output.to_f64_vec().expect("output should be f64");
        assert!((vals[0] - 1.0).abs() < 1e-10, "1^2 = 1");
        assert!((vals[1] - 16.0).abs() < 1e-10, "4^2 = 16");
        assert!((vals[2] - 81.0).abs() < 1e-10, "9^2 = 81");
    }
}
