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
    use fj_core::{CompatibilityMode, Jaxpr};
    use std::collections::BTreeMap;

    fn empty_jaxpr() -> Jaxpr {
        Jaxpr {
            invars: vec![],
            constvars: vec![],
            outvars: vec![],
            equations: vec![],
        }
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
}
