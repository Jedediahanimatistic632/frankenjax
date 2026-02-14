#![forbid(unsafe_code)]

pub mod durability;

use fj_core::{
    CompatibilityMode, ProgramSpec, TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_jax_code/jax"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            strict_mode: true,
        }
    }

    #[must_use]
    pub fn mode(&self) -> CompatibilityMode {
        if self.strict_mode {
            CompatibilityMode::Strict
        } else {
            CompatibilityMode::Hardened
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureFamily {
    Jit,
    Grad,
    Vmap,
    Lax,
    Random,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureManifestEntry {
    pub family: FixtureFamily,
    pub path_pattern: String,
    pub representative_test: String,
}

#[must_use]
pub fn default_fixture_manifest() -> Vec<FixtureManifestEntry> {
    vec![
        FixtureManifestEntry {
            family: FixtureFamily::Jit,
            path_pattern: "tests/jax_jit_test.py".to_owned(),
            representative_test: "JaxJitTest".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Grad,
            path_pattern: "tests/lax_autodiff_test.py".to_owned(),
            representative_test: "LAX_GRAD_OPS".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Vmap,
            path_pattern: "tests/lax_vmap_test.py".to_owned(),
            representative_test: "LaxVmapTest::_CheckBatching".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Lax,
            path_pattern: "tests/lax_test.py".to_owned(),
            representative_test: "LaxTest::testOp".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Random,
            path_pattern: "tests/random_test.py".to_owned(),
            representative_test: "RandomValuesCase".to_owned(),
        },
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub manifest_family_count: usize,
    pub strict_mode: bool,
}

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = collect_json_fixtures(&config.fixture_root).len();
    let _mode = config.mode();

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        manifest_family_count: default_fixture_manifest().len(),
        strict_mode: config.strict_mode,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityCase {
    pub case_id: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityReport {
    pub total_cases: usize,
    pub matched_cases: usize,
    pub mismatched_cases: usize,
    pub strict_mode: bool,
}

#[must_use]
pub fn evaluate_parity(cases: &[ParityCase], strict_mode: bool) -> ParityReport {
    let matched_cases = cases
        .iter()
        .filter(|case| case.expected == case.actual)
        .count();

    ParityReport {
        total_cases: cases.len(),
        matched_cases,
        mismatched_cases: cases.len().saturating_sub(matched_cases),
        strict_mode,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureNote {
    pub suite: String,
    pub oracle: String,
    pub notes: String,
}

pub fn read_fixture_note(path: &Path) -> Result<FixtureNote, std::io::Error> {
    let raw = fs::read_to_string(path)?;
    let parsed = serde_json::from_str::<FixtureNote>(&raw).map_err(std::io::Error::other)?;
    Ok(parsed)
}

#[must_use]
pub fn collect_json_fixtures(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_json_fixtures_recursive(root, &mut out);
    out.sort();
    out
}

fn collect_json_fixtures_recursive(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json_fixtures_recursive(&path, out);
            continue;
        }

        if path.extension().is_some_and(|ext| ext == "json") {
            out.push(path);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureMode {
    Strict,
    Hardened,
}

impl FixtureMode {
    #[must_use]
    pub fn as_runtime_mode(self) -> CompatibilityMode {
        match self {
            Self::Strict => CompatibilityMode::Strict,
            Self::Hardened => CompatibilityMode::Hardened,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureProgram {
    Add2,
    Square,
    SquarePlusLinear,
    AddOne,
    SinX,
    CosX,
}

impl FixtureProgram {
    #[must_use]
    pub fn as_program_spec(self) -> ProgramSpec {
        match self {
            Self::Add2 => ProgramSpec::Add2,
            Self::Square => ProgramSpec::Square,
            Self::SquarePlusLinear => ProgramSpec::SquarePlusLinear,
            Self::AddOne => ProgramSpec::AddOne,
            Self::SinX => ProgramSpec::SinX,
            Self::CosX => ProgramSpec::CosX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureTransform {
    Jit,
    Grad,
    Vmap,
}

impl FixtureTransform {
    #[must_use]
    pub fn as_runtime_transform(self) -> Transform {
        match self {
            Self::Jit => Transform::Jit,
            Self::Grad => Transform::Grad,
            Self::Vmap => Transform::Vmap,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FixtureValue {
    ScalarF64 { value: f64 },
    ScalarI64 { value: i64 },
    VectorF64 { values: Vec<f64> },
    VectorI64 { values: Vec<i64> },
}

impl FixtureValue {
    pub fn to_runtime_value(&self) -> Result<Value, String> {
        match self {
            Self::ScalarF64 { value } => Ok(Value::scalar_f64(*value)),
            Self::ScalarI64 { value } => Ok(Value::scalar_i64(*value)),
            Self::VectorF64 { values } => Value::vector_f64(values)
                .map_err(|err| format!("vector_f64 conversion failed: {err}")),
            Self::VectorI64 { values } => Value::vector_i64(values)
                .map_err(|err| format!("vector_i64 conversion failed: {err}")),
        }
    }

    pub fn approx_matches(&self, actual: &Value, atol: f64, rtol: f64) -> bool {
        match self {
            Self::ScalarF64 { value } => actual
                .as_f64_scalar()
                .is_some_and(|actual_value| approx_equal(*value, actual_value, atol, rtol)),
            Self::ScalarI64 { value } => actual
                .as_scalar_literal()
                .and_then(fj_core::Literal::as_i64)
                .is_some_and(|actual_value| actual_value == *value),
            Self::VectorF64 { values } => actual
                .as_tensor()
                .and_then(|tensor| tensor.to_f64_vec())
                .is_some_and(|actual_values| {
                    values.len() == actual_values.len()
                        && values
                            .iter()
                            .zip(actual_values.iter())
                            .all(|(expected, actual)| approx_equal(*expected, *actual, atol, rtol))
                }),
            Self::VectorI64 { values } => actual.as_tensor().is_some_and(|tensor| {
                if tensor.elements.len() != values.len() {
                    return false;
                }
                tensor
                    .elements
                    .iter()
                    .zip(values.iter())
                    .all(|(actual, expected)| actual.as_i64().is_some_and(|v| v == *expected))
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformFixtureCase {
    pub case_id: String,
    pub family: FixtureFamily,
    pub mode: FixtureMode,
    pub program: FixtureProgram,
    pub transforms: Vec<FixtureTransform>,
    pub args: Vec<FixtureValue>,
    pub expected: Vec<FixtureValue>,
    pub atol: f64,
    pub rtol: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformFixtureBundle {
    pub schema_version: String,
    pub generated_by: String,
    pub generated_at_unix_ms: u128,
    pub cases: Vec<TransformFixtureCase>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformCaseReport {
    pub case_id: String,
    pub family: FixtureFamily,
    pub mode: FixtureMode,
    pub comparator: ComparatorKind,
    pub drift_classification: DriftClassification,
    pub matched: bool,
    pub expected_json: String,
    pub actual_json: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformParityReport {
    pub schema_version: String,
    pub total_cases: usize,
    pub matched_cases: usize,
    pub mismatched_cases: usize,
    pub reports: Vec<TransformCaseReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparatorKind {
    Exact,
    ApproxAtolRtol,
    ShapeOnly,
    TypeOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftClassification {
    Pass,
    Regression,
    Improvement,
    Flake,
    Timeout,
}

pub fn read_transform_fixture_bundle(
    path: &Path,
) -> Result<TransformFixtureBundle, std::io::Error> {
    let raw = fs::read_to_string(path)?;
    let parsed =
        serde_json::from_str::<TransformFixtureBundle>(&raw).map_err(std::io::Error::other)?;
    Ok(parsed)
}

#[must_use]
pub fn run_transform_fixture_bundle(
    _config: &HarnessConfig,
    bundle: &TransformFixtureBundle,
) -> TransformParityReport {
    let mut reports = Vec::with_capacity(bundle.cases.len());

    for case in &bundle.cases {
        reports.push(run_transform_fixture_case(case));
    }

    let matched_cases = reports.iter().filter(|report| report.matched).count();

    TransformParityReport {
        schema_version: "frankenjax.transform-parity-report.v1".to_owned(),
        total_cases: reports.len(),
        matched_cases,
        mismatched_cases: reports.len().saturating_sub(matched_cases),
        reports,
    }
}

#[must_use]
pub fn emit_parity_markdown(report: &TransformParityReport) -> String {
    let mut out = String::new();
    out.push_str("# Transform Parity Report\n\n");
    out.push_str("| Metric | Value |\n");
    out.push_str("|---|---|\n");
    out.push_str(&format!("| Schema | `{}` |\n", report.schema_version));
    out.push_str(&format!("| Total Cases | {} |\n", report.total_cases));
    out.push_str(&format!("| Matched Cases | {} |\n", report.matched_cases));
    out.push_str(&format!(
        "| Mismatched Cases | {} |\n\n",
        report.mismatched_cases
    ));

    out.push_str("| Case ID | Family | Mode | Comparator | Drift | Matched |\n");
    out.push_str("|---|---|---|---|---|---|\n");
    for case in &report.reports {
        out.push_str(&format!(
            "| {} | {:?} | {:?} | {:?} | {:?} | {} |\n",
            case.case_id,
            case.family,
            case.mode,
            case.comparator,
            case.drift_classification,
            case.matched
        ));
    }

    out
}

fn run_transform_fixture_case(case: &TransformFixtureCase) -> TransformCaseReport {
    let runtime_args = case
        .args
        .iter()
        .map(FixtureValue::to_runtime_value)
        .collect::<Result<Vec<_>, _>>();

    let expected_json = serde_json::to_string(&case.expected)
        .unwrap_or_else(|err| format!("<expected serialization error: {err}>"));

    let runtime_args = match runtime_args {
        Ok(args) => args,
        Err(err) => {
            return TransformCaseReport {
                case_id: case.case_id.clone(),
                family: case.family,
                mode: case.mode,
                comparator: ComparatorKind::ApproxAtolRtol,
                drift_classification: classify_drift(
                    false,
                    Some("fixture argument conversion failed"),
                ),
                matched: false,
                expected_json,
                actual_json: None,
                error: Some(format!("fixture argument conversion failed: {err}")),
            };
        }
    };

    let mut ledger = TraceTransformLedger::new(build_program(case.program.as_program_spec()));
    for (idx, transform) in case.transforms.iter().enumerate() {
        let runtime_transform = transform.as_runtime_transform();
        ledger.push_transform(
            runtime_transform,
            format!(
                "fixture:{}:{}:{}",
                case.case_id,
                runtime_transform.as_str(),
                idx
            ),
        );
    }

    let result = dispatch(DispatchRequest {
        mode: case.mode.as_runtime_mode(),
        ledger,
        args: runtime_args,
        backend: "cpu".to_owned(),
        compile_options: std::collections::BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: Vec::new(),
    });

    match result {
        Ok(response) => {
            let matched =
                response.outputs.len() == case.expected.len()
                    && case.expected.iter().zip(response.outputs.iter()).all(
                        |(expected, actual)| expected.approx_matches(actual, case.atol, case.rtol),
                    );

            let actual_json = serde_json::to_string(&response.outputs)
                .unwrap_or_else(|err| format!("<actual serialization error: {err}>"));

            TransformCaseReport {
                case_id: case.case_id.clone(),
                family: case.family,
                mode: case.mode,
                comparator: ComparatorKind::ApproxAtolRtol,
                drift_classification: classify_drift(
                    matched,
                    if matched {
                        None
                    } else {
                        Some("output mismatch")
                    },
                ),
                matched,
                expected_json,
                actual_json: Some(actual_json),
                error: if matched {
                    None
                } else {
                    Some("output mismatch".to_owned())
                },
            }
        }
        Err(err) => TransformCaseReport {
            case_id: case.case_id.clone(),
            family: case.family,
            mode: case.mode,
            comparator: ComparatorKind::ApproxAtolRtol,
            drift_classification: classify_drift(false, Some(&err.to_string())),
            matched: false,
            expected_json,
            actual_json: None,
            error: Some(err.to_string()),
        },
    }
}

fn approx_equal(expected: f64, actual: f64, atol: f64, rtol: f64) -> bool {
    let tolerance = atol + rtol * expected.abs();
    (expected - actual).abs() <= tolerance
}

fn classify_drift(matched: bool, error: Option<&str>) -> DriftClassification {
    if matched {
        return DriftClassification::Pass;
    }

    if error.is_some_and(|detail| detail.to_ascii_lowercase().contains("timeout")) {
        return DriftClassification::Timeout;
    }

    DriftClassification::Regression
}

#[cfg(test)]
mod tests {
    use super::{
        DriftClassification, FixtureFamily, FixtureMode, FixtureProgram, FixtureTransform,
        FixtureValue, HarnessConfig, ParityCase, TransformFixtureBundle, TransformFixtureCase,
        classify_drift, collect_json_fixtures, default_fixture_manifest, emit_parity_markdown,
        evaluate_parity, read_fixture_note, run_smoke, run_transform_fixture_bundle,
    };

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert_eq!(report.manifest_family_count, 5);
        assert!(report.strict_mode);
    }

    #[test]
    fn manifest_covers_core_families() {
        let manifest = default_fixture_manifest();
        assert_eq!(manifest.len(), 5);
    }

    #[test]
    fn parity_report_counts_matches() {
        let report = evaluate_parity(
            &[
                ParityCase {
                    case_id: "a".to_owned(),
                    expected: "1".to_owned(),
                    actual: "1".to_owned(),
                },
                ParityCase {
                    case_id: "b".to_owned(),
                    expected: "2".to_owned(),
                    actual: "3".to_owned(),
                },
            ],
            true,
        );

        assert_eq!(report.total_cases, 2);
        assert_eq!(report.matched_cases, 1);
        assert_eq!(report.mismatched_cases, 1);
        assert!(report.strict_mode);
    }

    #[test]
    fn fixture_note_round_trip() {
        let cfg = HarnessConfig::default_paths();
        let note_path = cfg.fixture_root.join("smoke_case.json");
        let note = read_fixture_note(&note_path).expect("fixture note should parse");
        assert_eq!(note.suite, "smoke");
    }

    #[test]
    fn fixture_collection_finds_json() {
        let cfg = HarnessConfig::default_paths();
        let fixtures = collect_json_fixtures(&cfg.fixture_root);
        assert!(!fixtures.is_empty());
    }

    #[test]
    fn transform_bundle_runs_and_matches() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                args: vec![
                    FixtureValue::ScalarI64 { value: 2 },
                    FixtureValue::ScalarI64 { value: 5 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 7 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };

        let report = run_transform_fixture_bundle(&cfg, &bundle);
        assert_eq!(report.total_cases, 1);
        assert_eq!(report.matched_cases, 1);
        assert_eq!(report.mismatched_cases, 0);
    }

    #[test]
    fn test_conformance_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("conformance", "smoke")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_conformance_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    #[test]
    fn test_drift_classification_timeout_and_regression() {
        assert_eq!(
            classify_drift(false, Some("timeout waiting for oracle")),
            DriftClassification::Timeout
        );
        assert_eq!(
            classify_drift(false, Some("shape mismatch")),
            DriftClassification::Regression
        );
        assert_eq!(classify_drift(true, None), DriftClassification::Pass);
    }

    #[test]
    fn test_parity_markdown_emitter_includes_summary() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar_markdown".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 2 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 3 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };
        let report = run_transform_fixture_bundle(&cfg, &bundle);
        let markdown = emit_parity_markdown(&report);
        assert!(markdown.contains("Transform Parity Report"));
        assert!(markdown.contains("jit_add_scalar_markdown"));
    }
}
