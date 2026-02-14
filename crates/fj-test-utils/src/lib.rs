#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const TEST_LOG_SCHEMA_VERSION: &str = "frankenjax.test-log.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestResult {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestLogEnv {
    pub rust_version: String,
    pub os: String,
    pub cargo_target_dir: String,
    pub timestamp_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TestPhaseTimings {
    pub setup_ms: u64,
    pub execute_ms: u64,
    pub verify_ms: u64,
    pub teardown_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestLogV1 {
    pub schema_version: String,
    pub test_id: String,
    pub fixture_id: String,
    pub seed: Option<u64>,
    pub mode: TestMode,
    pub env: TestLogEnv,
    pub artifact_refs: Vec<String>,
    pub result: TestResult,
    pub duration_ms: u64,
    pub details: Option<String>,
    pub phase_timings: TestPhaseTimings,
}

impl TestLogV1 {
    #[must_use]
    pub fn unit(
        test_id: impl Into<String>,
        fixture_id: impl Into<String>,
        mode: TestMode,
        result: TestResult,
    ) -> Self {
        Self {
            schema_version: TEST_LOG_SCHEMA_VERSION.to_owned(),
            test_id: test_id.into(),
            fixture_id: fixture_id.into(),
            seed: capture_proptest_seed(),
            mode,
            env: capture_env(),
            artifact_refs: Vec::new(),
            result,
            duration_ms: 0,
            details: None,
            phase_timings: TestPhaseTimings::default(),
        }
    }
}

#[must_use]
pub fn capture_env() -> TestLogEnv {
    TestLogEnv {
        rust_version: rust_version(),
        os: std::env::consts::OS.to_owned(),
        cargo_target_dir: std::env::var("CARGO_TARGET_DIR")
            .unwrap_or_else(|_| "<default>".to_owned()),
        timestamp_unix_ms: now_unix_ms_u64(),
    }
}

pub fn fixture_id_from_json<T: Serialize>(fixture: &T) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(fixture)?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

#[must_use]
pub fn property_test_case_count() -> u32 {
    if let Ok(raw) = std::env::var("FJ_PROPTEST_CASES")
        && let Ok(parsed) = raw.parse::<u32>()
        && parsed > 0
    {
        return parsed;
    }

    if std::env::var_os("CI").is_some() {
        1024
    } else {
        256
    }
}

#[must_use]
pub fn capture_proptest_seed() -> Option<u64> {
    if let Ok(raw) = std::env::var("FJ_PROPTEST_SEED")
        && let Ok(seed) = raw.parse::<u64>()
    {
        return Some(seed);
    }

    if let Ok(raw) = std::env::var("PROPTEST_RNG_SEED")
        && let Ok(seed) = raw.parse::<u64>()
    {
        return Some(seed);
    }

    None
}

#[must_use]
pub fn test_id(module_path: &str, test_name: &str) -> String {
    format!("{module_path}::{test_name}")
}

fn now_unix_ms_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

fn rust_version() -> String {
    let output = Command::new("rustc").arg("--version").output();
    match output {
        Ok(result) if result.status.success() => {
            String::from_utf8_lossy(&result.stdout).trim().to_owned()
        }
        _ => "rustc <unknown>".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TEST_LOG_SCHEMA_VERSION, TestLogV1, TestMode, TestResult, fixture_id_from_json,
        property_test_case_count, test_id,
    };

    #[test]
    fn test_fixture_digest_deterministic_json() {
        let fixture = serde_json::json!({
            "a": 1,
            "b": [2, 3, 4]
        });
        let digest_a = fixture_id_from_json(&fixture).expect("digest should build");
        let digest_b = fixture_id_from_json(&fixture).expect("digest should build");
        assert_eq!(digest_a, digest_b);
        assert_eq!(digest_a.len(), 64);
    }

    #[test]
    fn test_property_case_count_has_default_floor() {
        assert!(property_test_case_count() >= 256);
    }

    #[test]
    fn test_log_schema_round_trip_serialization() {
        let log = TestLogV1::unit(
            test_id(module_path!(), "test_log_schema_round_trip_serialization"),
            "fixture-id",
            TestMode::Strict,
            TestResult::Pass,
        );
        assert_eq!(log.schema_version, TEST_LOG_SCHEMA_VERSION);
        let encoded = serde_json::to_string(&log).expect("serialize should work");
        let decoded: TestLogV1 = serde_json::from_str(&encoded).expect("deserialize should work");
        assert_eq!(decoded.schema_version, TEST_LOG_SCHEMA_VERSION);
    }
}
