# Reliability Gate Artifacts

This directory stores machine-readable outputs for reliability gates.
Per-run gate logs and diagnostics live under `ci-artifacts/<run-id>/`.

## Budget Source

- `reliability_budgets.v1.json` — normative gate budgets (runtime/perf/crash thresholds and related knobs).
- `coverage_trend.v1.json` — append-only coverage trend snapshots used for regression checks.
- `flake_quarantine_policy.md` — quarantine and re-promotion procedure.
- `github_actions_reliability_gates.example.yml` — CI wiring template.

## Generated Reports

- `crash_report.v1.json` — open/new `P0` crash triage output used by gate `G4`.
- `reliability_gate_report.v1.json` — aggregated ordered `G1..G8` gate status + failures.
- `runs/<run-id>/manifest.json` — indexed run manifest generated from gate report and artifact scans.

## Commands

- `./scripts/enforce_quality_gates.sh`
- `./scripts/enforce_quality_gates.sh --run-id local-dev --skip-g6 --skip-g8`
- `./scripts/generate_run_manifest.sh --run-id <run-id> --output ci-artifacts/<run-id>`
