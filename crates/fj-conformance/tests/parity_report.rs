//! Tests for V1 Parity Report generation (spec Section 8.3)

use fj_conformance::{
    ComparatorKind, DriftClassification, FamilyCaseEntry, FixtureFamily, FixtureMode,
    ParityReportV1, TransformCaseReport, TransformParityReport,
};

fn make_case(
    id: &str,
    family: FixtureFamily,
    matched: bool,
    error: Option<&str>,
) -> TransformCaseReport {
    TransformCaseReport {
        case_id: id.to_owned(),
        family,
        mode: FixtureMode::Strict,
        comparator: ComparatorKind::ApproxAtolRtol,
        drift_classification: if matched {
            DriftClassification::Pass
        } else {
            DriftClassification::Regression
        },
        matched,
        expected_json: r#"{"scalar_f64":1.0}"#.to_owned(),
        actual_json: if matched {
            Some(r#"{"scalar_f64":1.0}"#.to_owned())
        } else {
            Some(r#"{"scalar_f64":999.0}"#.to_owned())
        },
        error: error.map(str::to_owned),
    }
}

fn make_sample_report() -> TransformParityReport {
    let reports = vec![
        make_case("jit_add_01", FixtureFamily::Jit, true, None),
        make_case("jit_mul_02", FixtureFamily::Jit, true, None),
        make_case("grad_square_01", FixtureFamily::Grad, true, None),
        make_case(
            "grad_fail_02",
            FixtureFamily::Grad,
            false,
            Some("tolerance exceeded"),
        ),
        make_case("vmap_add_01", FixtureFamily::Vmap, true, None),
        make_case("lax_sin_01", FixtureFamily::Lax, true, None),
        make_case("lax_cos_02", FixtureFamily::Lax, true, None),
        make_case("lax_exp_03", FixtureFamily::Lax, true, None),
    ];
    TransformParityReport {
        schema_version: "frankenjax.transform-parity.v1".to_owned(),
        total_cases: reports.len(),
        matched_cases: reports.iter().filter(|r| r.matched).count(),
        mismatched_cases: reports.iter().filter(|r| !r.matched).count(),
        reports,
    }
}

#[test]
fn test_parity_report_schema() {
    let transform_report = make_sample_report();
    let v1 =
        ParityReportV1::from_transform_report(&transform_report, "strict", "0.1.0", "jax-0.9.0.1");
    let json_str = v1.to_json().expect("should serialize");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");

    // Verify spec Section 8.3 fields
    assert_eq!(
        parsed["schema_version"], "frankenjax.parity-report.v1",
        "schema_version"
    );
    assert!(parsed["timestamp"].is_string(), "timestamp present");
    assert_eq!(parsed["fj_version"], "0.1.0");
    assert_eq!(parsed["oracle_version"], "jax-0.9.0.1");
    assert_eq!(parsed["mode"], "strict");
    assert!(parsed["families"].is_object(), "families is object");
    assert!(parsed["summary"].is_object(), "summary is object");
    assert!(parsed["summary"]["total"].is_number(), "summary.total");
    assert!(
        parsed["summary"]["pass_rate"].is_number(),
        "summary.pass_rate"
    );
    assert!(parsed["summary"]["gate"].is_string(), "summary.gate");
}

#[test]
fn test_parity_report_total_cases() {
    let transform_report = make_sample_report();
    let v1 =
        ParityReportV1::from_transform_report(&transform_report, "strict", "0.1.0", "jax-0.9.0.1");

    assert_eq!(v1.summary.total, 8, "total should count all fixtures");

    // Per-family totals should sum to total
    let family_total: usize = v1.families.values().map(|f| f.total).sum();
    assert_eq!(family_total, v1.summary.total);
}

#[test]
fn test_parity_report_matched_count() {
    let transform_report = make_sample_report();
    let v1 =
        ParityReportV1::from_transform_report(&transform_report, "strict", "0.1.0", "jax-0.9.0.1");

    // 7 matched out of 8
    let matched_count: usize = v1.families.values().map(|f| f.matched).sum();
    assert_eq!(matched_count, 7);
    assert_eq!(v1.summary.pass_rate, 7.0 / 8.0);
}

#[test]
fn test_parity_report_mismatched_detail() {
    let transform_report = make_sample_report();
    let v1 =
        ParityReportV1::from_transform_report(&transform_report, "strict", "0.1.0", "jax-0.9.0.1");

    let grad_family = v1.families.get("grad").expect("grad family should exist");
    let mismatched: Vec<&FamilyCaseEntry> =
        grad_family.cases.iter().filter(|c| !c.matched).collect();
    assert_eq!(mismatched.len(), 1);
    assert_eq!(mismatched[0].case_id, "grad_fail_02");
    assert!(
        mismatched[0].expected.is_some(),
        "mismatched should include expected"
    );
    assert!(
        mismatched[0].actual.is_some(),
        "mismatched should include actual"
    );
    assert_eq!(
        mismatched[0].error.as_deref(),
        Some("tolerance exceeded"),
        "error should be preserved"
    );
}

#[test]
fn test_parity_report_per_family_breakdown() {
    let transform_report = make_sample_report();
    let v1 =
        ParityReportV1::from_transform_report(&transform_report, "strict", "0.1.0", "jax-0.9.0.1");

    // Check all 4 families present
    assert!(v1.families.contains_key("jit"), "jit family");
    assert!(v1.families.contains_key("grad"), "grad family");
    assert!(v1.families.contains_key("vmap"), "vmap family");
    assert!(v1.families.contains_key("lax"), "lax family");

    // Check per-family counts
    let jit = &v1.families["jit"];
    assert_eq!(jit.total, 2);
    assert_eq!(jit.matched, 2);
    assert_eq!(jit.mismatched, 0);

    let grad = &v1.families["grad"];
    assert_eq!(grad.total, 2);
    assert_eq!(grad.matched, 1);
    assert_eq!(grad.mismatched, 1);

    let vmap = &v1.families["vmap"];
    assert_eq!(vmap.total, 1);
    assert_eq!(vmap.matched, 1);

    let lax = &v1.families["lax"];
    assert_eq!(lax.total, 3);
    assert_eq!(lax.matched, 3);
}

#[test]
fn test_parity_report_mode_field() {
    let transform_report = make_sample_report();

    let strict =
        ParityReportV1::from_transform_report(&transform_report, "strict", "0.1.0", "jax-0.9.0.1");
    assert_eq!(strict.mode, "strict");

    let hardened = ParityReportV1::from_transform_report(
        &transform_report,
        "hardened",
        "0.1.0",
        "jax-0.9.0.1",
    );
    assert_eq!(hardened.mode, "hardened");
}

#[test]
fn test_parity_report_metadata() {
    let transform_report = make_sample_report();
    let v1 =
        ParityReportV1::from_transform_report(&transform_report, "strict", "0.1.0", "jax-0.9.0.1");

    assert_eq!(v1.fj_version, "0.1.0");
    assert_eq!(v1.oracle_version, "jax-0.9.0.1");
    assert!(!v1.timestamp.is_empty(), "timestamp should be non-empty");
    assert_eq!(v1.schema_version, "frankenjax.parity-report.v1");
}

#[test]
fn test_parity_report_ci_integration() {
    // All matched → gate passes
    let all_pass_reports = vec![
        make_case("case_01", FixtureFamily::Jit, true, None),
        make_case("case_02", FixtureFamily::Grad, true, None),
    ];
    let all_pass = TransformParityReport {
        schema_version: "frankenjax.transform-parity.v1".to_owned(),
        total_cases: 2,
        matched_cases: 2,
        mismatched_cases: 0,
        reports: all_pass_reports,
    };
    let v1_pass =
        ParityReportV1::from_transform_report(&all_pass, "strict", "0.1.0", "jax-0.9.0.1");
    assert!(v1_pass.gate_passes(), "all matched should pass gate");
    assert_eq!(v1_pass.ci_exit_code(), 0, "CI should exit 0");

    // Some mismatched → gate fails
    let some_fail = make_sample_report(); // has 1 mismatch
    let v1_fail =
        ParityReportV1::from_transform_report(&some_fail, "strict", "0.1.0", "jax-0.9.0.1");
    assert!(!v1_fail.gate_passes(), "mismatch should fail gate");
    assert_eq!(v1_fail.ci_exit_code(), 1, "CI should exit 1");

    // Markdown output is parseable
    let md = v1_fail.to_markdown();
    assert!(md.contains("# Parity Report V1"));
    assert!(md.contains("Pass Rate"));
    assert!(md.contains("Per-Family Breakdown"));
}
