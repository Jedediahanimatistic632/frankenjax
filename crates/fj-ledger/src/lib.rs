#![forbid(unsafe_code)]

use fj_core::CompatibilityMode;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionAction {
    Keep,
    Kill,
    Reprofile,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSignal {
    pub signal_name: String,
    pub log_likelihood_delta: f64,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossMatrix {
    pub keep_if_useful: u32,
    pub kill_if_useful: u32,
    pub keep_if_abandoned: u32,
    pub kill_if_abandoned: u32,
}

impl Default for LossMatrix {
    fn default() -> Self {
        Self {
            keep_if_useful: 0,
            kill_if_useful: 100,
            keep_if_abandoned: 30,
            kill_if_abandoned: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub mode: CompatibilityMode,
    pub posterior_abandoned: f64,
    pub expected_loss_keep: f64,
    pub expected_loss_kill: f64,
    pub action: DecisionAction,
    pub timestamp_unix_ms: u128,
}

impl DecisionRecord {
    #[must_use]
    pub fn from_posterior(
        mode: CompatibilityMode,
        posterior_abandoned: f64,
        matrix: &LossMatrix,
    ) -> Self {
        let expected_loss_keep = expected_loss_keep(posterior_abandoned, matrix);
        let expected_loss_kill = expected_loss_kill(posterior_abandoned, matrix);
        let action = if expected_loss_keep < expected_loss_kill {
            DecisionAction::Keep
        } else if expected_loss_kill < expected_loss_keep {
            DecisionAction::Kill
        } else {
            DecisionAction::Reprofile
        };

        Self {
            mode,
            posterior_abandoned,
            expected_loss_keep,
            expected_loss_kill,
            action,
            timestamp_unix_ms: now_unix_ms(),
        }
    }
}

#[must_use]
pub fn recommend_action(posterior_abandoned: f64, matrix: &LossMatrix) -> DecisionAction {
    DecisionRecord::from_posterior(CompatibilityMode::Strict, posterior_abandoned, matrix).action
}

#[must_use]
pub fn expected_loss_keep(posterior_abandoned: f64, matrix: &LossMatrix) -> f64 {
    let useful_prob = 1.0 - posterior_abandoned;
    useful_prob * f64::from(matrix.keep_if_useful)
        + posterior_abandoned * f64::from(matrix.keep_if_abandoned)
}

#[must_use]
pub fn expected_loss_kill(posterior_abandoned: f64, matrix: &LossMatrix) -> f64 {
    let useful_prob = 1.0 - posterior_abandoned;
    useful_prob * f64::from(matrix.kill_if_useful)
        + posterior_abandoned * f64::from(matrix.kill_if_abandoned)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub decision_id: String,
    pub record: DecisionRecord,
    pub signals: Vec<EvidenceSignal>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvidenceLedger {
    entries: Vec<LedgerEntry>,
}

impl EvidenceLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn append(&mut self, entry: LedgerEntry) {
        self.entries.push(entry);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::{
        DecisionAction, DecisionRecord, EvidenceLedger, LossMatrix, expected_loss_keep,
        expected_loss_kill, recommend_action,
    };
    use fj_core::CompatibilityMode;

    #[test]
    fn loss_matrix_prefers_keep_when_useful_probability_is_high() {
        let matrix = LossMatrix::default();
        let keep = expected_loss_keep(0.1, &matrix);
        let kill = expected_loss_kill(0.1, &matrix);
        assert!(keep < kill);
        assert_eq!(recommend_action(0.1, &matrix), DecisionAction::Keep);
    }

    #[test]
    fn loss_matrix_prefers_kill_when_abandoned_probability_is_high() {
        let matrix = LossMatrix::default();
        let keep = expected_loss_keep(0.95, &matrix);
        let kill = expected_loss_kill(0.95, &matrix);
        assert!(kill < keep);
        assert_eq!(recommend_action(0.95, &matrix), DecisionAction::Kill);
    }

    #[test]
    fn decision_record_includes_timestamp() {
        let matrix = LossMatrix::default();
        let record = DecisionRecord::from_posterior(CompatibilityMode::Hardened, 0.5, &matrix);
        assert!(record.timestamp_unix_ms > 0);
    }

    #[test]
    fn ledger_append_increases_length() {
        let mut ledger = EvidenceLedger::new();
        assert!(ledger.is_empty());
        ledger.append(super::LedgerEntry {
            decision_id: "d1".to_owned(),
            record: DecisionRecord::from_posterior(
                CompatibilityMode::Strict,
                0.3,
                &LossMatrix::default(),
            ),
            signals: Vec::new(),
        });
        assert_eq!(ledger.len(), 1);
    }
}
