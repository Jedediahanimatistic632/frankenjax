# FrankenJAX Artifact Topology Lock (v1)

This lock defines canonical artifact families and schema bindings for Phase-2C packet execution.

## Canonical Families

| Artifact Family | Schema Version | Canonical Path Pattern |
|---|---|---|
| legacy anchor map | `frankenjax.legacy-anchor-map.v1` | `artifacts/phase2c/<packet_id>/legacy_anchor_map.v1.json` |
| contract table | `frankenjax.contract-table.v1` | `artifacts/phase2c/<packet_id>/contract_table.v1.json` |
| fixture manifest | `frankenjax.fixture-manifest.v1` | `artifacts/phase2c/<packet_id>/fixture_manifest.v1.json` |
| parity gate | `frankenjax.parity-gate.v1` | `artifacts/phase2c/<packet_id>/parity_gate.v1.json` |
| risk note | `frankenjax.risk-note.v1` | `artifacts/phase2c/<packet_id>/risk_note.v1.json` |
| compatibility matrix | `frankenjax.compatibility-matrix.v1` | `artifacts/phase2c/global/compatibility_matrix.v1.json` |
| test log | `frankenjax.test-log.v1` | `artifacts/testing/logs/<suite>/<test_id>.json` |

## Required Gate Behavior

- Missing required fields in any artifact => packet status `NOT READY`.
- Unknown `schema_version` in strict mode => fail closed.
- Hardened mode unknown-version handling requires explicit allowlist + audit entry.

## Validation Source of Truth

- Schemas: `artifacts/schemas/*.v1.schema.json`
- Valid examples: `artifacts/examples/*.v1.example.json`
- Invalid examples: `artifacts/examples/invalid/*.missing-required.example.json`
- Test gate: `crates/fj-conformance/tests/artifact_schemas.rs`
