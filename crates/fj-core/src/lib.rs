#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<u32>,
}

impl Shape {
    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    #[must_use]
    pub fn vector(len: u32) -> Self {
        Self { dims: vec![len] }
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[must_use]
    pub fn element_count(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1_u64, |acc, dim| acc.checked_mul(u64::from(*dim)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbstractValue {
    pub dtype: DType,
    pub shape: Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Primitive {
    Add,
    Mul,
    Dot,
    Sin,
    Cos,
    ReduceSum,
}

impl Primitive {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Dot => "dot",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::ReduceSum => "reduce_sum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Transform {
    Jit,
    Grad,
    Vmap,
}

impl Transform {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jit => "jit",
            Self::Grad => "grad",
            Self::Vmap => "vmap",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VarId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Literal {
    I64(i64),
    Bool(bool),
    F64Bits(u64),
}

impl Literal {
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self::F64Bits(value.to_bits())
    }

    #[must_use]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Self::F64Bits(bits) => Some(f64::from_bits(bits)),
            Self::I64(value) => Some(value as f64),
            Self::Bool(_) => None,
        }
    }

    #[must_use]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::I64(value) => Some(value),
            Self::Bool(_) | Self::F64Bits(_) => None,
        }
    }

    #[must_use]
    pub fn is_integral(self) -> bool {
        matches!(self, Self::I64(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Value {
    Scalar(Literal),
    Tensor(TensorValue),
}

impl Value {
    #[must_use]
    pub fn scalar_i64(value: i64) -> Self {
        Self::Scalar(Literal::I64(value))
    }

    #[must_use]
    pub fn scalar_f64(value: f64) -> Self {
        Self::Scalar(Literal::from_f64(value))
    }

    #[must_use]
    pub fn scalar_bool(value: bool) -> Self {
        Self::Scalar(Literal::Bool(value))
    }

    pub fn vector_i64(values: &[i64]) -> Result<Self, ValueError> {
        let elements = values.iter().copied().map(Literal::I64).collect::<Vec<_>>();
        Ok(Self::Tensor(TensorValue::new(
            DType::I64,
            Shape::vector(values.len() as u32),
            elements,
        )?))
    }

    pub fn vector_f64(values: &[f64]) -> Result<Self, ValueError> {
        let elements = values
            .iter()
            .copied()
            .map(Literal::from_f64)
            .collect::<Vec<_>>();
        Ok(Self::Tensor(TensorValue::new(
            DType::F64,
            Shape::vector(values.len() as u32),
            elements,
        )?))
    }

    #[must_use]
    pub fn as_scalar_literal(&self) -> Option<Literal> {
        match self {
            Self::Scalar(lit) => Some(*lit),
            Self::Tensor(_) => None,
        }
    }

    #[must_use]
    pub fn as_f64_scalar(&self) -> Option<f64> {
        self.as_scalar_literal().and_then(Literal::as_f64)
    }

    #[must_use]
    pub fn as_tensor(&self) -> Option<&TensorValue> {
        match self {
            Self::Scalar(_) => None,
            Self::Tensor(tensor) => Some(tensor),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorValue {
    pub dtype: DType,
    pub shape: Shape,
    pub elements: Vec<Literal>,
}

impl TensorValue {
    pub fn new(dtype: DType, shape: Shape, elements: Vec<Literal>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count as usize != elements.len() {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: elements.len(),
            });
        }

        Ok(Self {
            dtype,
            shape,
            elements,
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        self.elements.iter().copied().map(Literal::as_f64).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    ShapeOverflow {
        shape: Shape,
    },
    ElementCountMismatch {
        shape: Shape,
        expected_count: u64,
        actual_count: usize,
    },
}

impl std::fmt::Display for ValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeOverflow { shape } => {
                write!(f, "shape element count overflowed: {:?}", shape.dims)
            }
            Self::ElementCountMismatch {
                shape,
                expected_count,
                actual_count,
            } => {
                write!(
                    f,
                    "tensor element count mismatch for shape {:?}: expected {}, got {}",
                    shape.dims, expected_count, actual_count
                )
            }
        }
    }
}

impl std::error::Error for ValueError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Atom {
    Var(VarId),
    Lit(Literal),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Equation {
    pub primitive: Primitive,
    pub inputs: Vec<Atom>,
    pub outputs: Vec<VarId>,
    pub params: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Jaxpr {
    pub invars: Vec<VarId>,
    pub constvars: Vec<VarId>,
    pub outvars: Vec<VarId>,
    pub equations: Vec<Equation>,
}

impl Jaxpr {
    #[must_use]
    pub fn canonical_fingerprint(&self) -> String {
        let mut out = String::new();
        write_var_list(&mut out, "in", &self.invars);
        write_var_list(&mut out, "const", &self.constvars);
        write_var_list(&mut out, "out", &self.outvars);

        for eqn in &self.equations {
            let _ = write!(&mut out, "eqn:{}(", eqn.primitive.as_str());
            for atom in &eqn.inputs {
                write_atom(&mut out, atom);
                out.push(',');
            }
            out.push(')');
            out.push_str("->");
            for outvar in &eqn.outputs {
                let _ = write!(&mut out, "v{},", outvar.0);
            }
            out.push('{');
            for (key, value) in &eqn.params {
                let _ = write!(&mut out, "{key}={value};");
            }
            out.push('}');
            out.push('|');
        }

        out
    }
}

fn write_var_list(out: &mut String, label: &str, vars: &[VarId]) {
    let _ = write!(out, "{label}=[");
    for var in vars {
        let _ = write!(out, "v{},", var.0);
    }
    out.push(']');
}

fn write_atom(out: &mut String, atom: &Atom) {
    match atom {
        Atom::Var(var) => {
            let _ = write!(out, "v{}", var.0);
        }
        Atom::Lit(lit) => write_literal(out, *lit),
    }
}

fn write_literal(out: &mut String, lit: Literal) {
    match lit {
        Literal::I64(value) => {
            let _ = write!(out, "i64:{value}");
        }
        Literal::Bool(value) => {
            let _ = write!(out, "bool:{value}");
        }
        Literal::F64Bits(value) => {
            let _ = write!(out, "f64bits:{value}");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgramSpec {
    Add2,
    Square,
    SquarePlusLinear,
    AddOne,
}

#[must_use]
pub fn build_program(spec: ProgramSpec) -> Jaxpr {
    match spec {
        ProgramSpec::Add2 => Jaxpr {
            invars: vec![VarId(1), VarId(2)],
            constvars: vec![],
            outvars: vec![VarId(3)],
            equations: vec![Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: vec![VarId(3)],
                params: BTreeMap::new(),
            }],
        },
        ProgramSpec::Square => Jaxpr {
            invars: vec![VarId(1)],
            constvars: vec![],
            outvars: vec![VarId(2)],
            equations: vec![Equation {
                primitive: Primitive::Mul,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: vec![VarId(2)],
                params: BTreeMap::new(),
            }],
        },
        ProgramSpec::SquarePlusLinear => Jaxpr {
            invars: vec![VarId(1)],
            constvars: vec![],
            outvars: vec![VarId(4)],
            equations: vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: vec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(2))],
                    outputs: vec![VarId(3)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                    outputs: vec![VarId(4)],
                    params: BTreeMap::new(),
                },
            ],
        },
        ProgramSpec::AddOne => Jaxpr {
            invars: vec![VarId(1)],
            constvars: vec![],
            outvars: vec![VarId(2)],
            equations: vec![Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))],
                outputs: vec![VarId(2)],
                params: BTreeMap::new(),
            }],
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceTransformLedger {
    pub root_jaxpr: Jaxpr,
    pub transform_stack: Vec<Transform>,
    pub transform_evidence: Vec<String>,
}

impl TraceTransformLedger {
    #[must_use]
    pub fn new(root_jaxpr: Jaxpr) -> Self {
        Self {
            root_jaxpr,
            transform_stack: Vec::new(),
            transform_evidence: Vec::new(),
        }
    }

    pub fn push_transform(&mut self, transform: Transform, evidence_id: impl Into<String>) {
        self.transform_stack.push(transform);
        self.transform_evidence.push(evidence_id.into());
    }

    #[must_use]
    pub fn composition_signature(&self) -> String {
        let mut out = String::new();
        out.push_str("stack=");
        for transform in &self.transform_stack {
            let _ = write!(&mut out, "{}>", transform.as_str());
        }
        out.push_str("|jaxpr=");
        out.push_str(&self.root_jaxpr.canonical_fingerprint());
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformCompositionProof {
    pub stack_signature: String,
    pub stack_hash_hex: String,
    pub transform_count: usize,
    pub evidence_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformCompositionError {
    EvidenceCountMismatch {
        transform_count: usize,
        evidence_count: usize,
    },
    EmptyEvidence {
        index: usize,
        transform: Transform,
    },
    UnsupportedSequence {
        detail: String,
    },
}

impl std::fmt::Display for TransformCompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvidenceCountMismatch {
                transform_count,
                evidence_count,
            } => {
                write!(
                    f,
                    "transform/evidence cardinality mismatch: transforms={}, evidence={}",
                    transform_count, evidence_count
                )
            }
            Self::EmptyEvidence { index, transform } => {
                write!(
                    f,
                    "transform evidence at index {} for {} is empty",
                    index,
                    transform.as_str()
                )
            }
            Self::UnsupportedSequence { detail } => {
                write!(f, "unsupported transform sequence: {detail}")
            }
        }
    }
}

impl std::error::Error for TransformCompositionError {}

pub fn verify_transform_composition(
    ledger: &TraceTransformLedger,
) -> Result<TransformCompositionProof, TransformCompositionError> {
    if ledger.transform_stack.len() != ledger.transform_evidence.len() {
        return Err(TransformCompositionError::EvidenceCountMismatch {
            transform_count: ledger.transform_stack.len(),
            evidence_count: ledger.transform_evidence.len(),
        });
    }

    for (index, transform) in ledger.transform_stack.iter().enumerate() {
        if ledger.transform_evidence[index].trim().is_empty() {
            return Err(TransformCompositionError::EmptyEvidence {
                index,
                transform: *transform,
            });
        }
    }

    let grad_count = ledger
        .transform_stack
        .iter()
        .filter(|transform| **transform == Transform::Grad)
        .count();
    if grad_count > 1 {
        return Err(TransformCompositionError::UnsupportedSequence {
            detail: "current engine supports at most one grad transform".to_owned(),
        });
    }

    let vmap_count = ledger
        .transform_stack
        .iter()
        .filter(|transform| **transform == Transform::Vmap)
        .count();
    if vmap_count > 1 {
        return Err(TransformCompositionError::UnsupportedSequence {
            detail: "current engine supports at most one vmap transform".to_owned(),
        });
    }

    let stack_signature = ledger.composition_signature();
    let stack_hash_hex = format!("{:016x}", fnv1a_64(stack_signature.as_bytes()));

    Ok(TransformCompositionProof {
        stack_signature,
        stack_hash_hex,
        transform_count: ledger.transform_stack.len(),
        evidence_count: ledger.transform_evidence.len(),
    })
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{
        Literal, ProgramSpec, Shape, TensorValue, TraceTransformLedger, Transform, Value,
        ValueError, build_program, verify_transform_composition,
    };

    #[test]
    fn shape_scalar_rank_is_zero() {
        let scalar = Shape::scalar();
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.element_count(), Some(1));
    }

    #[test]
    fn vector_constructor_builds_tensor_value() {
        let value = Value::vector_i64(&[1, 2, 3]).expect("vector constructor should succeed");
        let tensor = value.as_tensor().expect("expected tensor value");
        assert_eq!(tensor.shape, Shape::vector(3));
        assert_eq!(tensor.len(), 3);
    }

    #[test]
    fn tensor_new_rejects_wrong_element_count() {
        let err = TensorValue::new(
            super::DType::I64,
            Shape::vector(4),
            vec![Literal::I64(1), Literal::I64(2)],
        )
        .expect_err("shape mismatch should fail");

        assert!(matches!(err, ValueError::ElementCountMismatch { .. }));
    }

    #[test]
    fn composition_proof_is_deterministic() {
        let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
        ttl.push_transform(Transform::Jit, "jit-evidence");
        ttl.push_transform(Transform::Grad, "grad-evidence");

        let proof_a = verify_transform_composition(&ttl).expect("proof should validate");
        let proof_b = verify_transform_composition(&ttl).expect("proof should validate");
        assert_eq!(proof_a.stack_hash_hex, proof_b.stack_hash_hex);
    }

    #[test]
    fn unsupported_sequence_rejected() {
        let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
        ttl.push_transform(Transform::Grad, "g1");
        ttl.push_transform(Transform::Grad, "g2");

        let err = verify_transform_composition(&ttl).expect_err("double grad should fail");
        assert!(format!("{err}").contains("at most one grad"));
    }
}
