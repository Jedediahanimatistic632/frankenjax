# Flake Quarantine Policy

## Detection

1. Weekly job runs `./scripts/detect_flakes.sh --runs 10`.
2. Any run with mixed pass/fail outcomes marks the suite as flaky.
3. Flake report is persisted at `artifacts/ci/flake_report.v1.json`.

## Classification

- `P0`: flake blocks safety or compatibility-critical tests.
- `P1`: flake in packet conformance/differential tests.
- `P2`: flake in non-critical tests.

## Quarantine Workflow

1. Open a bead for the flaky test/suite with log references.
2. Move flaky test into a quarantine test target/module.
3. Keep quarantine tests runnable but non-blocking until fixed.
4. Re-promotion requires 100 consecutive pass runs (`--runs 100`).

## Promotion Gate

- Re-enable blocking mode only after:
  - deterministic pass streak (100/100),
  - root-cause note attached,
  - no new flakes detected in next weekly run.
