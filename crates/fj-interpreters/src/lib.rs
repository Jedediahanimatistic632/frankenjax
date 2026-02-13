#![forbid(unsafe_code)]

use fj_core::{Atom, Jaxpr, Value, VarId};
use fj_lax::{EvalError, eval_primitive};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpreterError {
    InputArity {
        expected: usize,
        actual: usize,
    },
    MissingVariable(VarId),
    UnexpectedOutputArity {
        primitive: fj_core::Primitive,
        actual: usize,
    },
    Primitive(EvalError),
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputArity { expected, actual } => {
                write!(
                    f,
                    "input arity mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::MissingVariable(var) => write!(f, "missing variable v{}", var.0),
            Self::UnexpectedOutputArity { primitive, actual } => write!(
                f,
                "expected single-output primitive {}, got {} outputs",
                primitive.as_str(),
                actual
            ),
            Self::Primitive(err) => write!(f, "primitive eval failed: {err}"),
        }
    }
}

impl std::error::Error for InterpreterError {}

impl From<EvalError> for InterpreterError {
    fn from(value: EvalError) -> Self {
        Self::Primitive(value)
    }
}

pub fn eval_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<Value>, InterpreterError> {
    if args.len() != jaxpr.invars.len() {
        return Err(InterpreterError::InputArity {
            expected: jaxpr.invars.len(),
            actual: args.len(),
        });
    }

    let mut env: BTreeMap<VarId, Value> = BTreeMap::new();
    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[idx].clone());
    }

    for eqn in &jaxpr.equations {
        if eqn.outputs.len() != 1 {
            return Err(InterpreterError::UnexpectedOutputArity {
                primitive: eqn.primitive,
                actual: eqn.outputs.len(),
            });
        }

        let mut resolved = Vec::with_capacity(eqn.inputs.len());
        for atom in &eqn.inputs {
            match atom {
                Atom::Var(var) => {
                    let value = env
                        .get(var)
                        .cloned()
                        .ok_or(InterpreterError::MissingVariable(*var))?;
                    resolved.push(value);
                }
                Atom::Lit(lit) => resolved.push(Value::Scalar(*lit)),
            }
        }

        let output = eval_primitive(eqn.primitive, &resolved)?;
        env.insert(eqn.outputs[0], output);
    }

    jaxpr
        .outvars
        .iter()
        .map(|var| {
            env.get(var)
                .cloned()
                .ok_or(InterpreterError::MissingVariable(*var))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{InterpreterError, eval_jaxpr};
    use fj_core::{ProgramSpec, Value, build_program};

    #[test]
    fn eval_simple_add_jaxpr() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let outputs = eval_jaxpr(&jaxpr, &[Value::scalar_i64(4), Value::scalar_i64(5)]);
        assert_eq!(outputs, Ok(vec![Value::scalar_i64(9)]));
    }

    #[test]
    fn eval_vector_add_one_jaxpr() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let output = eval_jaxpr(
            &jaxpr,
            &[Value::vector_i64(&[1, 2, 3]).expect("vector value should build")],
        )
        .expect("vector add should succeed");

        assert_eq!(
            output,
            vec![Value::vector_i64(&[2, 3, 4]).expect("vector value should build")]
        );
    }

    #[test]
    fn input_arity_mismatch_is_reported() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let err = eval_jaxpr(&jaxpr, &[Value::scalar_i64(4)]).expect_err("should fail");
        assert_eq!(
            err,
            InterpreterError::InputArity {
                expected: 2,
                actual: 1,
            }
        );
    }
}
