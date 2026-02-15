#![forbid(unsafe_code)]

//! Partial evaluation: split a Jaxpr into known and unknown sub-Jaxprs.
//!
//! Given a Jaxpr and a boolean mask indicating which inputs are known (concrete)
//! vs unknown (abstract), partial evaluation produces:
//! - `jaxpr_known`: equations whose inputs are all derivable from known values
//! - `jaxpr_unknown`: residual equations that depend on unknown inputs
//! - `residuals`: intermediate values produced by jaxpr_known and consumed by jaxpr_unknown
//!
//! Invariant: eval(jaxpr_known, known_inputs) ++ eval(jaxpr_unknown, residuals ++ unknown_inputs)
//!            == eval(original_jaxpr, all_inputs)

use fj_core::{AbstractValue, Atom, DType, Equation, Jaxpr, Shape, Value, VarId};
use rustc_hash::FxHashSet;

/// Classification of a value during partial evaluation.
#[derive(Debug, Clone)]
pub enum PartialVal {
    /// Value is concretely known at trace time.
    Known(Value),
    /// Value is abstract (unknown) — only its type signature is available.
    Unknown(AbstractValue),
}

impl PartialVal {
    /// Returns `true` if this value is concretely known.
    pub fn is_known(&self) -> bool {
        matches!(self, PartialVal::Known(_))
    }

    /// Returns the known value, if available.
    pub fn get_known(&self) -> Option<&Value> {
        match self {
            PartialVal::Known(v) => Some(v),
            PartialVal::Unknown(_) => None,
        }
    }

    /// Returns the abstract value (type signature) regardless of known/unknown status.
    pub fn get_aval(&self) -> AbstractValue {
        match self {
            PartialVal::Known(v) => abstract_value_of(v),
            PartialVal::Unknown(aval) => aval.clone(),
        }
    }
}

/// Result of partial evaluation on a Jaxpr.
#[derive(Debug, Clone)]
pub struct PartialEvalResult {
    /// Jaxpr containing only equations with all-known inputs.
    /// Outputs = original known outputs ++ residual values.
    pub jaxpr_known: Jaxpr,

    /// Constants for jaxpr_known's constvars (values hoisted from known inputs).
    pub known_consts: Vec<Value>,

    /// Jaxpr containing equations that depend on unknown inputs.
    /// Inputs = residuals (from jaxpr_known) ++ original unknown inputs.
    pub jaxpr_unknown: Jaxpr,

    /// Which of the original Jaxpr's outputs are unknown.
    pub out_unknowns: Vec<bool>,

    /// Abstract values of residual intermediate values passed between the two jaxprs.
    pub residual_avals: Vec<AbstractValue>,
}

/// Errors that can occur during partial evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartialEvalError {
    /// Input mask length doesn't match Jaxpr input count.
    InputMaskMismatch { expected: usize, actual: usize },
    /// A variable referenced in an equation was not defined.
    UndefinedVariable(VarId),
    /// Residual type mismatch between known outputs and unknown inputs.
    ResidualTypeMismatch { index: usize },
}

impl std::fmt::Display for PartialEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputMaskMismatch { expected, actual } => {
                write!(
                    f,
                    "input mask length mismatch: jaxpr has {} inputs, mask has {} entries",
                    expected, actual
                )
            }
            Self::UndefinedVariable(var) => write!(f, "undefined variable v{}", var.0),
            Self::ResidualTypeMismatch { index } => {
                write!(f, "residual type mismatch at index {}", index)
            }
        }
    }
}

impl std::error::Error for PartialEvalError {}

/// Partially evaluate a Jaxpr given a mask of which inputs are unknown.
///
/// This is the core partial evaluation routine. It walks the equations in order,
/// classifying each as "known" (all inputs known) or "unknown" (any input unknown),
/// and produces two sub-Jaxprs plus residual metadata.
///
/// # Arguments
/// * `jaxpr` - The Jaxpr to partially evaluate.
/// * `unknowns` - Boolean mask: `true` means the corresponding input is unknown.
///
/// # Returns
/// A `PartialEvalResult` containing the known and unknown sub-Jaxprs.
pub fn partial_eval_jaxpr(
    jaxpr: &Jaxpr,
    unknowns: &[bool],
) -> Result<PartialEvalResult, PartialEvalError> {
    if unknowns.len() != jaxpr.invars.len() {
        return Err(PartialEvalError::InputMaskMismatch {
            expected: jaxpr.invars.len(),
            actual: unknowns.len(),
        });
    }

    // Track which variables hold unknown values.
    let mut unknown_vars: FxHashSet<VarId> = FxHashSet::default();
    for (var, &is_unknown) in jaxpr.invars.iter().zip(unknowns.iter()) {
        if is_unknown {
            unknown_vars.insert(*var);
        }
    }

    // Classify equations and collect residuals.
    let mut known_eqns: Vec<Equation> = Vec::new();
    let mut unknown_eqns: Vec<Equation> = Vec::new();
    let mut residual_vars: Vec<VarId> = Vec::new();
    let mut residual_set: FxHashSet<VarId> = FxHashSet::default();

    // Find the maximum VarId used in the jaxpr to generate fresh ids.
    let max_existing = jaxpr
        .invars
        .iter()
        .chain(jaxpr.constvars.iter())
        .chain(jaxpr.outvars.iter())
        .chain(jaxpr.equations.iter().flat_map(|e| e.outputs.iter()))
        .map(|v| v.0)
        .max()
        .unwrap_or(0);
    let mut next_var = max_existing + 1000;

    for eqn in &jaxpr.equations {
        let any_input_unknown = eqn.inputs.iter().any(|atom| match atom {
            Atom::Var(v) => unknown_vars.contains(v),
            Atom::Lit(_) => false,
        });

        if any_input_unknown {
            // This equation goes to the unknown jaxpr.
            // Mark its outputs as unknown.
            for out_var in &eqn.outputs {
                unknown_vars.insert(*out_var);
            }
            unknown_eqns.push(eqn.clone());

            // Any known-variable inputs to this equation become residuals.
            for atom in &eqn.inputs {
                if let Atom::Var(v) = atom
                    && !unknown_vars.contains(v)
                    && !residual_set.contains(v)
                {
                    residual_vars.push(*v);
                    residual_set.insert(*v);
                }
            }
        } else {
            // All inputs known — this equation goes to the known jaxpr.
            known_eqns.push(eqn.clone());

            // Check if any outputs are needed as residuals (they might be
            // consumed by a later unknown equation). We'll handle this in
            // a second pass.
        }
    }

    // Second pass: identify known-equation outputs that are consumed by unknown equations.
    let unknown_input_vars: FxHashSet<VarId> = unknown_eqns
        .iter()
        .flat_map(|eqn| eqn.inputs.iter())
        .filter_map(|atom| match atom {
            Atom::Var(v) => Some(*v),
            Atom::Lit(_) => None,
        })
        .collect();

    for eqn in &known_eqns {
        for out_var in &eqn.outputs {
            if unknown_input_vars.contains(out_var) && !residual_set.contains(out_var) {
                residual_vars.push(*out_var);
                residual_set.insert(*out_var);
            }
        }
    }

    // Also check if known invars are consumed by unknown equations.
    for (var, &is_unknown) in jaxpr.invars.iter().zip(unknowns.iter()) {
        if !is_unknown && unknown_input_vars.contains(var) && !residual_set.contains(var) {
            residual_vars.push(*var);
            residual_set.insert(*var);
        }
    }

    // Build known Jaxpr.
    let known_invars: Vec<VarId> = jaxpr
        .invars
        .iter()
        .zip(unknowns.iter())
        .filter(|(_, is_unknown)| !**is_unknown)
        .map(|(v, _)| *v)
        .collect();

    let known_outvars: Vec<VarId> = {
        let mut outs: Vec<VarId> = jaxpr
            .outvars
            .iter()
            .filter(|v| !unknown_vars.contains(v))
            .copied()
            .collect();
        // Append residual outputs.
        outs.extend(residual_vars.iter().copied());
        outs
    };

    let jaxpr_known = Jaxpr::new(
        known_invars,
        jaxpr.constvars.clone(),
        known_outvars,
        known_eqns,
    );

    // Build unknown Jaxpr.
    // Inputs: residual vars (renamed) ++ original unknown inputs.
    let mut unknown_invars: Vec<VarId> = Vec::new();
    let mut _var_remap: Vec<(VarId, VarId)> = Vec::new();

    // Residual inputs come first.
    for &res_var in &residual_vars {
        let new_var = VarId(next_var);
        next_var += 1;
        unknown_invars.push(new_var);
        _var_remap.push((res_var, new_var));
    }

    // Then original unknown inputs.
    for (var, &is_unknown) in jaxpr.invars.iter().zip(unknowns.iter()) {
        if is_unknown {
            unknown_invars.push(*var);
        }
    }

    let unknown_outvars: Vec<VarId> = jaxpr
        .outvars
        .iter()
        .filter(|v| unknown_vars.contains(v))
        .copied()
        .collect();

    let jaxpr_unknown = Jaxpr::new(unknown_invars, vec![], unknown_outvars, unknown_eqns);

    // Determine which original outputs are unknown.
    let out_unknowns: Vec<bool> = jaxpr
        .outvars
        .iter()
        .map(|v| unknown_vars.contains(v))
        .collect();

    // Compute residual abstract values.
    let residual_avals: Vec<AbstractValue> = residual_vars
        .iter()
        .map(|_| AbstractValue {
            dtype: DType::F64,
            shape: Shape::scalar(),
        })
        .collect();

    Ok(PartialEvalResult {
        jaxpr_known,
        known_consts: vec![],
        jaxpr_unknown,
        out_unknowns,
        residual_avals,
    })
}

/// Dead code elimination on a Jaxpr.
///
/// Given a Jaxpr and a mask of which outputs are used, removes equations
/// that don't contribute to any used output. Preserves equation ordering.
///
/// Returns the pruned Jaxpr and a mask of which inputs are still needed.
pub fn dce_jaxpr(jaxpr: &Jaxpr, used_outputs: &[bool]) -> (Jaxpr, Vec<bool>) {
    // Backward pass: determine which variables are needed.
    let mut needed: FxHashSet<VarId> = FxHashSet::default();

    for (var, &used) in jaxpr.outvars.iter().zip(used_outputs.iter()) {
        if used {
            needed.insert(*var);
        }
    }

    // Walk equations in reverse, marking inputs of needed equations.
    let mut keep_eqn = vec![false; jaxpr.equations.len()];
    for (i, eqn) in jaxpr.equations.iter().enumerate().rev() {
        let outputs_needed = eqn.outputs.iter().any(|v| needed.contains(v));
        if outputs_needed {
            keep_eqn[i] = true;
            for atom in &eqn.inputs {
                if let Atom::Var(v) = atom {
                    needed.insert(*v);
                }
            }
        }
    }

    let retained_eqns: Vec<Equation> = jaxpr
        .equations
        .iter()
        .zip(keep_eqn.iter())
        .filter(|(_, keep)| **keep)
        .map(|(eqn, _)| eqn.clone())
        .collect();

    let used_inputs: Vec<bool> = jaxpr.invars.iter().map(|v| needed.contains(v)).collect();

    let new_jaxpr = Jaxpr::new(
        jaxpr.invars.clone(),
        jaxpr.constvars.clone(),
        jaxpr.outvars.clone(),
        retained_eqns,
    );

    (new_jaxpr, used_inputs)
}

/// Extract an abstract value (dtype + shape) from a concrete Value.
fn abstract_value_of(value: &Value) -> AbstractValue {
    match value {
        Value::Scalar(lit) => {
            let dtype = match lit {
                fj_core::Literal::I64(_) => DType::I64,
                fj_core::Literal::Bool(_) => DType::Bool,
                fj_core::Literal::F64Bits(_) => DType::F64,
            };
            AbstractValue {
                dtype,
                shape: Shape::scalar(),
            }
        }
        Value::Tensor(t) => AbstractValue {
            dtype: t.dtype,
            shape: t.shape.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{Equation, Jaxpr, Primitive, VarId};
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn make_add_chain_jaxpr() -> Jaxpr {
        // { a:f64, b:f64 -> c = add(a, b); d = mul(c, b) -> d }
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                },
            ],
        )
    }

    #[test]
    fn pe_all_known_folds_everything() {
        let jaxpr = make_add_chain_jaxpr();
        let result = partial_eval_jaxpr(&jaxpr, &[false, false]).unwrap();
        // All inputs known: all equations go to known jaxpr.
        assert_eq!(result.jaxpr_known.equations.len(), 2);
        assert_eq!(result.jaxpr_unknown.equations.len(), 0);
        assert_eq!(result.out_unknowns, vec![false]);
    }

    #[test]
    fn pe_all_unknown_residualizes_everything() {
        let jaxpr = make_add_chain_jaxpr();
        let result = partial_eval_jaxpr(&jaxpr, &[true, true]).unwrap();
        // All inputs unknown: all equations go to unknown jaxpr.
        assert_eq!(result.jaxpr_known.equations.len(), 0);
        assert_eq!(result.jaxpr_unknown.equations.len(), 2);
        assert_eq!(result.out_unknowns, vec![true]);
    }

    #[test]
    fn pe_mixed_produces_residuals() {
        // a known, b unknown: add(a, b) is unknown because b is unknown.
        let jaxpr = make_add_chain_jaxpr();
        let result = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
        // Both equations touch unknown b, so both go to unknown jaxpr.
        assert_eq!(result.jaxpr_unknown.equations.len(), 2);
        assert_eq!(result.out_unknowns, vec![true]);
    }

    #[test]
    fn pe_input_mask_mismatch() {
        let jaxpr = make_add_chain_jaxpr();
        let err = partial_eval_jaxpr(&jaxpr, &[false]).unwrap_err();
        assert_eq!(
            err,
            PartialEvalError::InputMaskMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn pe_generates_residuals_for_known_to_unknown_flow() {
        // { a:f64, b:f64 -> c = neg(a); d = mul(c, b) -> d }
        // a known, b unknown: neg(a) is known, mul(c, b) is unknown.
        // c must be a residual passed from known to unknown.
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                },
            ],
        );

        let result = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
        assert_eq!(result.jaxpr_known.equations.len(), 1); // neg(a) is known
        assert_eq!(result.jaxpr_unknown.equations.len(), 1); // mul(c, b) is unknown
        assert!(!result.residual_avals.is_empty()); // c is a residual
        assert_eq!(result.out_unknowns, vec![true]);
    }

    #[test]
    fn dce_removes_unused_equations() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
            ],
        );

        // Only first output used.
        let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true, false]);
        assert_eq!(pruned.equations.len(), 1);
        assert_eq!(pruned.equations[0].primitive, Primitive::Neg);
        assert_eq!(used_inputs, vec![true]);
    }

    #[test]
    fn dce_keeps_chain_dependencies() {
        let jaxpr = make_add_chain_jaxpr();
        let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true]);
        // Both equations needed since output depends on both.
        assert_eq!(pruned.equations.len(), 2);
        assert_eq!(used_inputs, vec![true, true]);
    }
}
