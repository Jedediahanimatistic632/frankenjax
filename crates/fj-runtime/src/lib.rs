#![forbid(unsafe_code)]

pub mod backend;
pub mod buffer;
pub mod device;

use fj_core::CompatibilityMode;
use fj_ledger::{DecisionAction, LossMatrix, recommend_action};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAdmissionModel {
    pub mode: CompatibilityMode,
    pub loss_matrix: LossMatrix,
}

impl RuntimeAdmissionModel {
    #[must_use]
    pub fn new(mode: CompatibilityMode) -> Self {
        Self {
            mode,
            loss_matrix: LossMatrix::default(),
        }
    }

    #[must_use]
    pub fn decide(&self, posterior_abandoned: f64) -> DecisionAction {
        let _mode = self.mode;
        recommend_action(posterior_abandoned, &self.loss_matrix)
    }
}

#[cfg(feature = "asupersync-integration")]
pub mod asupersync_bridge {
    use asupersync::{Cx, Error};

    pub fn emit_checkpoint(cx: &Cx, message: impl Into<String>) -> Result<(), Error> {
        cx.checkpoint_with(message.into())
    }

    #[must_use]
    pub fn cancellation_requested(cx: &Cx) -> bool {
        cx.is_cancel_requested()
    }
}

#[cfg(feature = "frankentui-integration")]
pub mod frankentui_bridge {
    use std::fmt::Write;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StatusCard {
        pub title: String,
        pub mode: String,
        pub confidence_percent: u8,
    }

    pub fn render_status_card(card: &StatusCard) -> String {
        let _type_check = std::any::type_name::<ftui::Style>();

        let mut out = String::new();
        let _ = writeln!(&mut out, "[{}]", card.title);
        let _ = writeln!(&mut out, "mode: {}", card.mode);
        let _ = write!(&mut out, "confidence: {}%", card.confidence_percent);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeAdmissionModel;
    use fj_core::CompatibilityMode;
    use fj_ledger::DecisionAction;

    #[test]
    fn runtime_admission_is_conservative_at_low_abandoned_probability() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        assert_eq!(model.decide(0.1), DecisionAction::Keep);
    }

    #[test]
    fn runtime_admission_kills_at_high_abandoned_probability() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        assert_eq!(model.decide(0.95), DecisionAction::Kill);
    }

    #[test]
    fn test_runtime_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("runtime", "admission")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_runtime_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }
}
