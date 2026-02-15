#![forbid(unsafe_code)]

//! Staging pipeline: trace → partial_eval → eval.
//!
//! Provides `make_jaxpr` for standalone Jaxpr construction from program specs,
//! and `StagedProgram` for the staging pipeline used by JIT.

use fj_core::{Jaxpr, ProgramSpec, Value, build_program};

use crate::partial_eval::{PartialEvalError, PartialEvalResult, partial_eval_jaxpr};
use crate::{InterpreterError, eval_jaxpr, eval_jaxpr_with_consts};

/// A staged program ready for execution with partial evaluation applied.
#[derive(Debug, Clone)]
pub struct StagedProgram {
    /// The known sub-Jaxpr (evaluated once with known inputs).
    pub jaxpr_known: Jaxpr,

    /// Constants for the known Jaxpr's constvars.
    pub known_consts: Vec<Value>,

    /// The unknown (residual) sub-Jaxpr (evaluated per call with dynamic inputs).
    pub jaxpr_unknown: Jaxpr,

    /// Which original outputs are produced by the unknown jaxpr.
    pub out_unknowns: Vec<bool>,

    /// Pre-computed residual values (outputs of evaluating jaxpr_known).
    pub residuals: Option<Vec<Value>>,
}

/// Errors during staging.
#[derive(Debug, Clone)]
pub enum StagingError {
    /// Partial evaluation failed.
    PartialEval(PartialEvalError),
    /// Known-jaxpr evaluation failed.
    KnownEval(InterpreterError),
    /// Unknown-jaxpr evaluation failed.
    UnknownEval(InterpreterError),
}

impl std::fmt::Display for StagingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PartialEval(e) => write!(f, "staging: partial eval failed: {e}"),
            Self::KnownEval(e) => write!(f, "staging: known eval failed: {e}"),
            Self::UnknownEval(e) => write!(f, "staging: unknown eval failed: {e}"),
        }
    }
}

impl std::error::Error for StagingError {}

/// Construct a Jaxpr from a program spec (FrankenJAX equivalent of `jax.make_jaxpr`).
///
/// Returns a valid Jaxpr with correct input/output types.
pub fn make_jaxpr(spec: ProgramSpec) -> Jaxpr {
    build_program(spec)
}

/// Stage a Jaxpr for execution with known/unknown input classification.
///
/// This implements the staging pipeline:
/// 1. Partial evaluate the Jaxpr to split known/unknown
/// 2. Evaluate the known sub-Jaxpr to get residuals
/// 3. Return a StagedProgram ready for dynamic execution
///
/// # Arguments
/// * `jaxpr` - The Jaxpr to stage.
/// * `unknowns` - Boolean mask of which inputs are unknown.
/// * `known_values` - Concrete values for the known inputs.
pub fn stage_jaxpr(
    jaxpr: &Jaxpr,
    unknowns: &[bool],
    known_values: &[Value],
) -> Result<StagedProgram, StagingError> {
    let pe_result: PartialEvalResult =
        partial_eval_jaxpr(jaxpr, unknowns).map_err(StagingError::PartialEval)?;

    // Evaluate the known sub-Jaxpr with the known inputs to produce residuals.
    let residuals = if !pe_result.jaxpr_known.equations.is_empty() {
        let known_outputs = eval_jaxpr_with_consts(
            &pe_result.jaxpr_known,
            &pe_result.known_consts,
            known_values,
        )
        .map_err(StagingError::KnownEval)?;
        Some(known_outputs)
    } else {
        None
    };

    Ok(StagedProgram {
        jaxpr_known: pe_result.jaxpr_known,
        known_consts: pe_result.known_consts,
        jaxpr_unknown: pe_result.jaxpr_unknown,
        out_unknowns: pe_result.out_unknowns,
        residuals,
    })
}

/// Execute a staged program with dynamic (unknown) inputs.
///
/// # Arguments
/// * `staged` - The staged program from `stage_jaxpr`.
/// * `dynamic_args` - Concrete values for the originally-unknown inputs.
pub fn execute_staged(
    staged: &StagedProgram,
    dynamic_args: &[Value],
) -> Result<Vec<Value>, StagingError> {
    if staged.jaxpr_unknown.equations.is_empty() {
        // All outputs were known — return residuals directly.
        return staged.residuals.clone().ok_or(StagingError::KnownEval(
            InterpreterError::InputArity {
                expected: 0,
                actual: 0,
            },
        ));
    }

    // Build inputs for the unknown jaxpr: residuals ++ dynamic_args.
    let mut unknown_inputs: Vec<Value> = Vec::new();
    if let Some(ref residuals) = staged.residuals {
        unknown_inputs.extend(residuals.iter().cloned());
    }
    unknown_inputs.extend(dynamic_args.iter().cloned());

    eval_jaxpr(&staged.jaxpr_unknown, &unknown_inputs).map_err(StagingError::UnknownEval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::ProgramSpec;

    #[test]
    fn make_jaxpr_produces_valid_programs() {
        let specs = [
            ProgramSpec::Add2,
            ProgramSpec::Square,
            ProgramSpec::AddOne,
            ProgramSpec::SinX,
            ProgramSpec::CosX,
        ];
        for spec in specs {
            let jaxpr = make_jaxpr(spec);
            assert!(!jaxpr.invars.is_empty() || !jaxpr.equations.is_empty());
        }
    }

    #[test]
    fn staging_all_known_produces_direct_result() {
        let jaxpr = make_jaxpr(ProgramSpec::Add2);
        let staged = stage_jaxpr(
            &jaxpr,
            &[false, false],
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
        )
        .unwrap();

        assert!(staged.jaxpr_unknown.equations.is_empty());
        assert!(staged.residuals.is_some());
    }
}
