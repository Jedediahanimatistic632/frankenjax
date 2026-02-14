# Reliability Gate Artifacts

This directory stores machine-readable outputs for reliability gates.

## Budget Source

- `reliability_budgets.v1.json` — normative line/branch coverage floors, flake budgets, and runtime budgets.
- `coverage_trend.v1.json` — append-only coverage trend snapshots used for regression checks.
- `flake_quarantine_policy.md` — quarantine and re-promotion procedure.
- `github_actions_reliability_gates.example.yml` — CI wiring template.

## Generated Reports

- `coverage_raw_llvmcov.v1.json` — raw `cargo llvm-cov` summary JSON.
- `coverage_report.v1.json` — per-crate floor checks and pass/fail outcomes.
- `flake_report.v1.json` — repeated-run flake detection output.
- `runtime_report.v1.json` — runtime budget checks for key suites.
- `reliability_gate_report.v1.json` — aggregated gate status.

## Commands

- `./scripts/enforce_quality_gates.sh`
- `./scripts/detect_flakes.sh --runs 10`
