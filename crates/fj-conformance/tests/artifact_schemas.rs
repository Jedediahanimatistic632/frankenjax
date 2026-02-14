#![forbid(unsafe_code)]

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_json(path: &Path) -> Value {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()))
}

fn validate_schema_examples(name: &str) {
    let root = repo_root();
    let schema_path = root.join(format!("artifacts/schemas/{name}.schema.json"));
    let valid_path = root.join(format!("artifacts/examples/{name}.example.json"));
    let invalid_path = root.join(format!(
        "artifacts/examples/invalid/{name}.missing-required.example.json"
    ));

    let schema = read_json(&schema_path);
    let validator = jsonschema::validator_for(&schema)
        .unwrap_or_else(|err| panic!("schema {} failed to compile: {err}", schema_path.display()));

    let valid_instance = read_json(&valid_path);
    let valid_errors = validator
        .iter_errors(&valid_instance)
        .map(|err| err.to_string())
        .collect::<Vec<_>>();
    assert!(
        valid_errors.is_empty(),
        "expected valid example {} to pass validation, got errors: {}",
        valid_path.display(),
        valid_errors.join(" | ")
    );

    let invalid_instance = read_json(&invalid_path);
    let invalid_errors = validator
        .iter_errors(&invalid_instance)
        .map(|err| err.to_string())
        .collect::<Vec<_>>();
    assert!(
        !invalid_errors.is_empty(),
        "expected invalid example {} to fail validation",
        invalid_path.display()
    );
}

#[test]
fn all_v1_artifact_schemas_have_valid_and_invalid_examples() {
    let schemas = [
        "legacy_anchor_map.v1",
        "contract_table.v1",
        "fixture_manifest.v1",
        "parity_gate.v1",
        "risk_note.v1",
        "compatibility_matrix.v1",
        "test_log.v1",
    ];

    for schema_name in schemas {
        validate_schema_examples(schema_name);
    }
}
