use std::path::Path;

use fj_conformance::{HarnessConfig, read_transform_fixture_bundle, run_transform_fixture_bundle};

#[test]
fn transform_fixture_bundle_matches_current_engine() {
    let cfg = HarnessConfig::default_paths();
    let fixture_path = cfg
        .fixture_root
        .join("transforms")
        .join("legacy_transform_cases.v1.json");
    assert!(
        Path::new(&fixture_path).exists(),
        "expected fixture bundle at {}",
        fixture_path.display()
    );

    let bundle = read_transform_fixture_bundle(&fixture_path)
        .expect("transform fixture bundle should parse");
    let report = run_transform_fixture_bundle(&cfg, &bundle);

    assert_eq!(report.total_cases, bundle.cases.len());
    assert_eq!(
        report.mismatched_cases, 0,
        "expected full parity for current transform fixture suite"
    );
}
