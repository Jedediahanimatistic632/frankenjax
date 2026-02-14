#![forbid(unsafe_code)]

use fj_core::{Atom, Jaxpr, Primitive, Value, VarId};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum AdError {
    UnsupportedPrimitive(Primitive),
    NonScalarGradientOutput,
    EvalFailed(String),
    MissingVariable(VarId),
    InputArity { expected: usize, actual: usize },
}

impl std::fmt::Display for AdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPrimitive(p) => {
                write!(f, "AD not implemented for primitive: {}", p.as_str())
            }
            Self::NonScalarGradientOutput => write!(f, "grad requires scalar output"),
            Self::EvalFailed(detail) => write!(f, "forward eval failed: {detail}"),
            Self::MissingVariable(var) => write!(f, "missing variable v{}", var.0),
            Self::InputArity { expected, actual } => {
                write!(f, "input arity mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for AdError {}

#[derive(Debug, Clone)]
struct TapeEntry {
    primitive: Primitive,
    inputs: Vec<VarId>,
    output: VarId,
    input_values: Vec<Value>,
}

#[derive(Debug, Clone)]
struct Tape {
    entries: Vec<TapeEntry>,
}

type ForwardResult = (Vec<Value>, Tape, BTreeMap<VarId, Value>);

fn forward_with_tape(jaxpr: &Jaxpr, args: &[Value]) -> Result<ForwardResult, AdError> {
    if args.len() != jaxpr.invars.len() {
        return Err(AdError::InputArity {
            expected: jaxpr.invars.len(),
            actual: args.len(),
        });
    }

    let mut env: BTreeMap<VarId, Value> = BTreeMap::new();
    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[idx].clone());
    }

    let mut tape = Tape {
        entries: Vec::with_capacity(jaxpr.equations.len()),
    };

    for eqn in &jaxpr.equations {
        let mut resolved = Vec::with_capacity(eqn.inputs.len());
        let mut input_var_ids = Vec::with_capacity(eqn.inputs.len());

        for atom in eqn.inputs.iter() {
            match atom {
                Atom::Var(var) => {
                    let value = env
                        .get(var)
                        .cloned()
                        .ok_or(AdError::MissingVariable(*var))?;
                    resolved.push(value);
                    input_var_ids.push(*var);
                }
                Atom::Lit(lit) => {
                    resolved.push(Value::Scalar(*lit));
                    // Literals get a sentinel VarId; their cotangent is discarded
                    input_var_ids.push(VarId(u32::MAX));
                }
            }
        }

        let output = eval_primitive(eqn.primitive, &resolved)
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

        let out_var = eqn.outputs[0];
        env.insert(out_var, output);

        tape.entries.push(TapeEntry {
            primitive: eqn.primitive,
            inputs: input_var_ids,
            output: out_var,
            input_values: resolved,
        });
    }

    let outputs = jaxpr
        .outvars
        .iter()
        .map(|var| env.get(var).cloned().ok_or(AdError::MissingVariable(*var)))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((outputs, tape, env))
}

fn backward(
    tape: &Tape,
    output_var: VarId,
    output_cotangent: f64,
    jaxpr: &Jaxpr,
) -> Result<Vec<f64>, AdError> {
    let mut adjoints: BTreeMap<VarId, f64> = BTreeMap::new();
    adjoints.insert(output_var, output_cotangent);

    for entry in tape.entries.iter().rev() {
        let g = adjoints.get(&entry.output).copied().unwrap_or(0.0);
        if g == 0.0 {
            continue;
        }

        let cotangents = vjp(entry.primitive, &entry.input_values, g)?;

        for (var_id, cot) in entry.inputs.iter().zip(cotangents.iter()) {
            if var_id.0 == u32::MAX {
                continue; // literal sentinel
            }
            *adjoints.entry(*var_id).or_insert(0.0) += cot;
        }
    }

    let grads = jaxpr
        .invars
        .iter()
        .map(|var| adjoints.get(var).copied().unwrap_or(0.0))
        .collect();

    Ok(grads)
}

fn vjp(primitive: Primitive, inputs: &[Value], g: f64) -> Result<Vec<f64>, AdError> {
    match primitive {
        Primitive::Add => Ok(vec![g, g]),
        Primitive::Mul => {
            let a = to_f64(&inputs[0])?;
            let b = to_f64(&inputs[1])?;
            Ok(vec![g * b, g * a])
        }
        Primitive::Sin => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![g * x.cos()])
        }
        Primitive::Cos => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![g * (-x.sin())])
        }
        Primitive::ReduceSum => {
            // For scalar input, cotangent is just g
            Ok(vec![g])
        }
        Primitive::Dot => {
            // rank-1 dot(a,b) = sum(a*b), cotangents: (g*b, g*a)
            let a = to_f64(&inputs[0])?;
            let b = to_f64(&inputs[1])?;
            Ok(vec![g * b, g * a])
        }
    }
}

fn to_f64(value: &Value) -> Result<f64, AdError> {
    value
        .as_f64_scalar()
        .ok_or(AdError::NonScalarGradientOutput)
}

/// Compute gradients of a Jaxpr with respect to all inputs.
///
/// The Jaxpr must produce a single scalar output.
pub fn grad_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<f64>, AdError> {
    let (outputs, tape, _env) = forward_with_tape(jaxpr, args)?;

    let output_val = outputs
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(AdError::NonScalarGradientOutput)?;
    let _ = output_val; // just verify it's scalar

    let output_var = jaxpr.outvars[0];
    backward(&tape, output_var, 1.0, jaxpr)
}

/// Compute gradient with respect to the first input only (convenience wrapper
/// matching JAX's default `grad` behavior).
pub fn grad_first(jaxpr: &Jaxpr, args: &[Value]) -> Result<f64, AdError> {
    let grads = grad_jaxpr(jaxpr, args)?;
    Ok(grads[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, build_program};

    #[test]
    fn grad_x_squared_at_3() {
        let jaxpr = build_program(ProgramSpec::Square);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(3.0)]).expect("grad should succeed");
        assert!(
            (grads[0] - 6.0).abs() < 1e-10,
            "d/dx(x²) at x=3 = 6, got {}",
            grads[0]
        );
    }

    #[test]
    fn grad_x_squared_plus_2x_at_3() {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(3.0)]).expect("grad should succeed");
        assert!(
            (grads[0] - 8.0).abs() < 1e-10,
            "d/dx(x²+2x) at x=3 = 8, got {}",
            grads[0]
        );
    }

    #[test]
    fn grad_sin_at_zero() {
        // Build sin(x) program: input x, output sin(x)
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;
        use std::collections::BTreeMap;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(0.0)]).expect("grad should succeed");
        assert!(
            (grads[0] - 1.0).abs() < 1e-10,
            "d/dx(sin(x)) at x=0 = cos(0) = 1, got {}",
            grads[0]
        );
    }

    #[test]
    fn grad_cos_at_zero() {
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;
        use std::collections::BTreeMap;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(0.0)]).expect("grad should succeed");
        assert!(
            grads[0].abs() < 1e-10,
            "d/dx(cos(x)) at x=0 = -sin(0) = 0, got {}",
            grads[0]
        );
    }

    #[test]
    fn symbolic_matches_numerical() {
        let jaxpr = build_program(ProgramSpec::Square);
        let x = 3.0;
        let symbolic = grad_first(&jaxpr, &[Value::scalar_f64(x)]).expect("symbolic grad");

        // Numerical (finite-diff)
        let eps = 1e-6;
        let plus = fj_lax::eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(x + eps), Value::scalar_f64(x + eps)],
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let minus = fj_lax::eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(x - eps), Value::scalar_f64(x - eps)],
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let numerical = (plus - minus) / (2.0 * eps);

        assert!(
            (symbolic - numerical).abs() < 1e-4,
            "symbolic={symbolic}, numerical={numerical}"
        );
    }

    #[test]
    fn test_ad_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&(3_u32, 9_u32)).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_ad_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    mod proptest_tests {
        use super::*;
        use fj_interpreters::eval_jaxpr;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]
            #[test]
            fn prop_ad_grad_square_matches_two_x(x in prop::num::f64::NORMAL.prop_filter(
                "finite and not too large",
                |x| x.is_finite() && x.abs() < 1e6
            )) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = build_program(ProgramSpec::Square);
                let symbolic = grad_first(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("symbolic grad");
                let expected = 2.0 * x;
                prop_assert!(
                    (symbolic - expected).abs() < 1e-8,
                    "grad(x²) at x={x}: symbolic={symbolic}, expected={expected}"
                );
            }

            #[test]
            fn prop_ad_symbolic_grad_matches_numerical(x in prop::num::f64::NORMAL.prop_filter(
                "finite and moderate",
                |x| x.is_finite() && x.abs() < 1e3
            )) {
                let jaxpr = build_program(ProgramSpec::Square);
                let symbolic = grad_first(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("symbolic grad");

                // Numerical finite-diff
                let eps = 1e-6;
                let plus = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x + eps)])
                    .unwrap()[0].as_f64_scalar().unwrap();
                let minus = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x - eps)])
                    .unwrap()[0].as_f64_scalar().unwrap();
                let numerical = (plus - minus) / (2.0 * eps);

                prop_assert!(
                    (symbolic - numerical).abs() < 1e-4,
                    "x={x}: symbolic={symbolic}, numerical={numerical}"
                );
            }
        }
    }
}
