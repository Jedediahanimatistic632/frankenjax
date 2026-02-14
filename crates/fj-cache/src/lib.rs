#![forbid(unsafe_code)]

use fj_core::{CompatibilityMode, Jaxpr, Transform};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKeyInput {
    pub mode: CompatibilityMode,
    pub backend: String,
    pub jaxpr: Jaxpr,
    pub transform_stack: Vec<Transform>,
    pub compile_options: BTreeMap<String, String>,
    pub custom_hook: Option<String>,
    pub unknown_incompatible_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    pub namespace: &'static str,
    pub digest_hex: String,
}

impl CacheKey {
    #[must_use]
    pub fn as_string(&self) -> String {
        format!("{}-{}", self.namespace, self.digest_hex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheKeyError {
    UnknownIncompatibleFeatures { features: Vec<String> },
}

impl std::fmt::Display for CacheKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownIncompatibleFeatures { features } => {
                write!(
                    f,
                    "strict mode rejected unknown incompatible features: {}",
                    features.join(",")
                )
            }
        }
    }
}

impl std::error::Error for CacheKeyError {}

pub fn build_cache_key(input: &CacheKeyInput) -> Result<CacheKey, CacheKeyError> {
    if input.mode == CompatibilityMode::Strict && !input.unknown_incompatible_features.is_empty() {
        return Err(CacheKeyError::UnknownIncompatibleFeatures {
            features: input.unknown_incompatible_features.clone(),
        });
    }

    let payload = canonical_payload(input);
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();

    Ok(CacheKey {
        namespace: "fjx",
        digest_hex: bytes_to_hex(&digest),
    })
}

#[must_use]
pub fn compatibility_matrix_row(input: &CacheKeyInput) -> String {
    format!(
        "mode={:?},backend={},unknown_features={}",
        input.mode,
        input.backend,
        input.unknown_incompatible_features.join(";")
    )
}

pub struct CacheKeyInputRef<'a> {
    pub mode: CompatibilityMode,
    pub backend: &'a str,
    pub jaxpr: &'a Jaxpr,
    pub transform_stack: &'a [Transform],
    pub compile_options: &'a BTreeMap<String, String>,
    pub custom_hook: Option<&'a str>,
    pub unknown_incompatible_features: &'a [String],
}

pub fn build_cache_key_ref(input: &CacheKeyInputRef<'_>) -> Result<CacheKey, CacheKeyError> {
    if input.mode == CompatibilityMode::Strict && !input.unknown_incompatible_features.is_empty() {
        return Err(CacheKeyError::UnknownIncompatibleFeatures {
            features: input.unknown_incompatible_features.to_vec(),
        });
    }

    let payload = canonical_payload_ref(input);
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();

    Ok(CacheKey {
        namespace: "fjx",
        digest_hex: bytes_to_hex(&digest),
    })
}

fn canonical_payload_ref(input: &CacheKeyInputRef<'_>) -> String {
    let transforms = input
        .transform_stack
        .iter()
        .map(|transform| transform.as_str())
        .collect::<Vec<_>>()
        .join(",");

    let compile_options = input
        .compile_options
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(";");

    let unknown = input.unknown_incompatible_features.join(",");

    format!(
        "mode={:?}|backend={}|transforms={}|compile={}|hook={}|unknown={}|jaxpr={}",
        input.mode,
        input.backend,
        transforms,
        compile_options,
        input.custom_hook.unwrap_or("none"),
        unknown,
        input.jaxpr.canonical_fingerprint(),
    )
}

fn canonical_payload(input: &CacheKeyInput) -> String {
    let transforms = input
        .transform_stack
        .iter()
        .map(|transform| transform.as_str())
        .collect::<Vec<_>>()
        .join(",");

    let compile_options = input
        .compile_options
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(";");

    let unknown = input.unknown_incompatible_features.join(",");

    format!(
        "mode={:?}|backend={}|transforms={}|compile={}|hook={}|unknown={}|jaxpr={}",
        input.mode,
        input.backend,
        transforms,
        compile_options,
        input.custom_hook.as_deref().unwrap_or("none"),
        unknown,
        input.jaxpr.canonical_fingerprint(),
    )
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = std::fmt::Write::write_fmt(&mut out, format_args!("{:02x}", byte));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{CacheKeyError, CacheKeyInput, build_cache_key};
    use fj_core::{CompatibilityMode, Jaxpr, Transform};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn empty_jaxpr() -> Jaxpr {
        Jaxpr::new(vec![], vec![], vec![], vec![])
    }

    #[test]
    fn strict_mode_rejects_unknown_features() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["mystery_field".to_owned()],
        };

        let err = build_cache_key(&input).expect_err("should reject unknown features");
        assert_eq!(
            err,
            CacheKeyError::UnknownIncompatibleFeatures {
                features: vec!["mystery_field".to_owned()],
            }
        );
    }

    #[test]
    fn hardened_mode_accepts_unknown_features() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Hardened,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["mystery_field".to_owned()],
        };

        let key = build_cache_key(&input).expect("hardened mode should hash unknown features");
        assert!(key.as_string().starts_with("fjx-"));
    }

    #[test]
    fn cache_key_is_stable_for_identical_inputs() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: Some("hook-a".to_owned()),
            unknown_incompatible_features: vec![],
        };

        let key_a = build_cache_key(&input).expect("key generation should succeed");
        let key_b = build_cache_key(&input).expect("key generation should succeed");
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn test_cache_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("cpu", "strict")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_cache_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    fn arb_transform() -> impl Strategy<Value = Transform> {
        prop_oneof![
            Just(Transform::Jit),
            Just(Transform::Grad),
            Just(Transform::Vmap),
        ]
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]
        #[test]
        fn prop_cache_key_stability(
            backend in "[a-z]{3,6}",
            transforms in proptest::collection::vec(arb_transform(), 0..3),
        ) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let input = CacheKeyInput {
                mode: CompatibilityMode::Hardened,
                backend,
                jaxpr: empty_jaxpr(),
                transform_stack: transforms,
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            };
            let key_a = build_cache_key(&input).expect("key gen");
            let key_b = build_cache_key(&input).expect("key gen");
            prop_assert_eq!(key_a, key_b);
        }
    }
}
