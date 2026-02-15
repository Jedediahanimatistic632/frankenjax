#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUDGETS_JSON="$ROOT_DIR/artifacts/ci/reliability_budgets.v1.json"
GATE_REPORT_JSON="$ROOT_DIR/artifacts/ci/reliability_gate_report.v1.json"
CRASH_REPORT_JSON="$ROOT_DIR/artifacts/ci/crash_report.v1.json"

RUN_ID=""
ARTIFACT_ROOT="$ROOT_DIR/ci-artifacts"

SKIP_G1=0
SKIP_G2=0
SKIP_G3=0
SKIP_G4=0
SKIP_G5=0
SKIP_G6=0
SKIP_G7=0
SKIP_G8=0

warn_legacy_skip_flake=0

usage() {
  cat <<'USAGE'
Usage: ./scripts/enforce_quality_gates.sh [options]

Ordered fail-fast gates:
  G1 = fmt/lint
  G2 = unit/property tests
  G3 = differential conformance
  G4 = adversarial/crash regression
  G5 = E2E scenarios
  G6 = performance regression
  G7 = artifact schema validation
  G8 = durability decode-proof verification

Options:
  --budgets <path>         Reliability budget JSON path.
  --run-id <id>            Explicit run id (default: timestamp-shortsha).
  --artifact-root <path>   Root directory for run artifacts (default: ./ci-artifacts).
  --skip-g1 ... --skip-g8  Skip specific gate(s).

Legacy option aliases (kept for compatibility):
  --skip-runtime           Alias for --skip-g2 --skip-g3 --skip-g5
  --skip-crash             Alias for --skip-g4
  --skip-perf              Alias for --skip-g6
  --skip-coverage          Alias for --skip-g7
  --skip-flake             No-op (flake is not a standalone G1..G8 gate)
  --flake-runs <n>         No-op (accepted for compatibility)

  -h, --help               Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --budgets)
      BUDGETS_JSON="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --skip-g1)
      SKIP_G1=1
      shift
      ;;
    --skip-g2)
      SKIP_G2=1
      shift
      ;;
    --skip-g3)
      SKIP_G3=1
      shift
      ;;
    --skip-g4)
      SKIP_G4=1
      shift
      ;;
    --skip-g5)
      SKIP_G5=1
      shift
      ;;
    --skip-g6)
      SKIP_G6=1
      shift
      ;;
    --skip-g7)
      SKIP_G7=1
      shift
      ;;
    --skip-g8)
      SKIP_G8=1
      shift
      ;;
    --skip-runtime)
      SKIP_G2=1
      SKIP_G3=1
      SKIP_G5=1
      shift
      ;;
    --skip-crash)
      SKIP_G4=1
      shift
      ;;
    --skip-perf)
      SKIP_G6=1
      shift
      ;;
    --skip-coverage)
      SKIP_G7=1
      shift
      ;;
    --skip-flake)
      warn_legacy_skip_flake=1
      shift
      ;;
    --flake-runs)
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $warn_legacy_skip_flake -eq 1 ]]; then
  echo "[warn] --skip-flake is a compatibility no-op in the G1..G8 pipeline." >&2
fi

if [[ ! -f "$BUDGETS_JSON" ]]; then
  echo "error: budgets file not found at $BUDGETS_JSON" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/artifacts/ci"
mkdir -p "$ARTIFACT_ROOT"

if [[ -z "$RUN_ID" ]]; then
  git_sha="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  RUN_ID="$(date +%Y%m%d-%H%M%S)-${git_sha}"
fi

RUN_DIR="$ARTIFACT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

USE_RCH=0
if command -v rch >/dev/null 2>&1 && [[ "${FJ_DISABLE_RCH:-0}" != "1" ]]; then
  USE_RCH=1
fi

now_ms() {
  date +%s%3N
}

run_cargo() {
  if [[ $USE_RCH -eq 1 ]]; then
    rch exec -- cargo "$@"
  else
    cargo "$@"
  fi
}

trim_summary() {
  local file="$1"
  local summary
  summary="$(tail -n 30 "$file" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
  if [[ -z "$summary" ]]; then
    summary="gate command failed"
  fi
  printf '%s' "${summary:0:200}"
}

gate_name() {
  case "$1" in
    G1) echo "fmt_lint" ;;
    G2) echo "unit_property_tests" ;;
    G3) echo "differential_conformance" ;;
    G4) echo "adversarial_crash_regression" ;;
    G5) echo "e2e_scenarios" ;;
    G6) echo "performance_regression" ;;
    G7) echo "artifact_schema_validation" ;;
    G8) echo "durability_decode_proof" ;;
    *) echo "unknown_gate" ;;
  esac
}

gate_replay_cmd() {
  case "$1" in
    G1) echo "cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings" ;;
    G2) echo "cargo test --workspace" ;;
    G3) echo "cargo test -p fj-conformance --test transforms -- --nocapture" ;;
    G4) echo "jq -s 'map(select(.severity==\"P0\" and ((.status // \"open\") != \"closed\") and ((.status // \"open\") != \"resolved\"))) | length' crates/fj-conformance/fuzz/corpus/crashes/index.v1.jsonl" ;;
    G5) echo "./scripts/run_e2e.sh" ;;
    G6) echo "./scripts/check_perf_regression.sh --budgets artifacts/ci/reliability_budgets.v1.json" ;;
    G7) echo "cargo test -p fj-conformance --test artifact_schemas -- --nocapture" ;;
    G8) echo "./scripts/durability_ci_gate.sh --skip-generate" ;;
    *) echo "echo unknown gate" ;;
  esac
}

gate_should_skip() {
  case "$1" in
    G1) [[ $SKIP_G1 -eq 1 ]] ;;
    G2) [[ $SKIP_G2 -eq 1 ]] ;;
    G3) [[ $SKIP_G3 -eq 1 ]] ;;
    G4) [[ $SKIP_G4 -eq 1 ]] ;;
    G5) [[ $SKIP_G5 -eq 1 ]] ;;
    G6) [[ $SKIP_G6 -eq 1 ]] ;;
    G7) [[ $SKIP_G7 -eq 1 ]] ;;
    G8) [[ $SKIP_G8 -eq 1 ]] ;;
    *) return 1 ;;
  esac
}

gate_g1() {
  run_cargo fmt --check
  run_cargo clippy --workspace --all-targets -- -D warnings
}

gate_g2() {
  run_cargo test --workspace --quiet
}

gate_g3() {
  run_cargo test -p fj-conformance --test transforms -- --nocapture
}

gate_g4() {
  local crash_index_rel crash_index crash_max_open_p0 crash_fail_on_new_p0
  local crash_known_p0_hashes_json open_p0_count new_open_p0_hashes_json
  local all_open_p0_json new_open_p0_count passed generated_at_ms

  crash_index_rel="$(jq -r '.crash_triage.index_path // "crates/fj-conformance/fuzz/corpus/crashes/index.v1.jsonl"' "$BUDGETS_JSON")"
  if [[ "$crash_index_rel" = /* ]]; then
    crash_index="$crash_index_rel"
  else
    crash_index="$ROOT_DIR/$crash_index_rel"
  fi

  crash_max_open_p0="$(jq -r '.crash_triage.max_open_p0 // 0' "$BUDGETS_JSON")"
  crash_fail_on_new_p0="$(jq -r '.crash_triage.fail_on_new_p0 // true' "$BUDGETS_JSON")"
  crash_known_p0_hashes_json="$(jq -c '.crash_triage.known_p0_hashes // []' "$BUDGETS_JSON")"

  if [[ -f "$crash_index" && -s "$crash_index" ]]; then
    open_p0_count="$(jq -s '
      map(
        select(
          .severity == "P0"
          and ((.status // "open") != "closed")
          and ((.status // "open") != "resolved")
        )
      ) | length
    ' "$crash_index")"

    new_open_p0_hashes_json="$(jq -s --argjson known "$crash_known_p0_hashes_json" '
      [
        .[]
        | select(
            .severity == "P0"
            and ((.status // "open") != "closed")
            and ((.status // "open") != "resolved")
          )
        | .crash_hash_sha256
        | select(. != null)
        | select(($known | index(.)) | not)
      ] | unique
    ' "$crash_index")"

    all_open_p0_json="$(jq -s '
      [
        .[]
        | select(
            .severity == "P0"
            and ((.status // "open") != "closed")
            and ((.status // "open") != "resolved")
          )
      ]
    ' "$crash_index")"
  else
    open_p0_count=0
    new_open_p0_hashes_json='[]'
    all_open_p0_json='[]'
  fi

  new_open_p0_count="$(jq -nr --argjson hashes "$new_open_p0_hashes_json" '$hashes | length')"
  passed=true

  if [[ "$open_p0_count" -gt "$crash_max_open_p0" ]]; then
    passed=false
    echo "open P0 count exceeded: open=${open_p0_count}, max=${crash_max_open_p0}" >&2
  fi
  if [[ "$crash_fail_on_new_p0" == "true" && "$new_open_p0_count" -gt 0 ]]; then
    passed=false
    echo "new open P0 crashes detected: count=${new_open_p0_count}" >&2
  fi

  generated_at_ms="$(now_ms)"
  jq -n \
    --arg schema_version "frankenjax.crash-report.v1" \
    --arg index_path "$crash_index" \
    --argjson index_exists "$( [[ -f "$crash_index" ]] && echo true || echo false )" \
    --argjson max_open_p0 "$crash_max_open_p0" \
    --argjson open_p0_count "$open_p0_count" \
    --argjson fail_on_new_p0 "$crash_fail_on_new_p0" \
    --argjson known_p0_hashes "$crash_known_p0_hashes_json" \
    --argjson new_open_p0_hashes "$new_open_p0_hashes_json" \
    --argjson open_p0_records "$all_open_p0_json" \
    --argjson passed "$passed" \
    --argjson generated_at_unix_ms "$generated_at_ms" \
    '{
      schema_version: $schema_version,
      generated_at_unix_ms: $generated_at_unix_ms,
      index_path: $index_path,
      index_exists: $index_exists,
      max_open_p0: $max_open_p0,
      open_p0_count: $open_p0_count,
      fail_on_new_p0: $fail_on_new_p0,
      known_p0_hashes: $known_p0_hashes,
      new_open_p0_hashes: $new_open_p0_hashes,
      open_p0_records: $open_p0_records,
      passed: $passed
    }' >"$CRASH_REPORT_JSON"

  [[ "$passed" == "true" ]]
}

gate_g5() {
  "$ROOT_DIR/scripts/run_e2e.sh"
}

gate_g6() {
  "$ROOT_DIR/scripts/check_perf_regression.sh" --budgets "$BUDGETS_JSON"
}

gate_g7() {
  run_cargo test -p fj-conformance --test artifact_schemas -- --nocapture
}

gate_g8() {
  "$ROOT_DIR/scripts/durability_ci_gate.sh" --skip-generate
}

gate_results_tmp="$(mktemp)"
failure_records_tmp="$(mktemp)"
trap 'rm -f "$gate_results_tmp" "$failure_records_tmp"' EXIT

append_gate_result() {
  local gate_id="$1"
  local name="$2"
  local status="$3"
  local duration_ms="$4"
  local failure_count="$5"
  jq -nc \
    --arg gate_id "$gate_id" \
    --arg name "$name" \
    --arg status "$status" \
    --argjson duration_ms "$duration_ms" \
    --argjson failure_count "$failure_count" \
    '{
      gate_id: $gate_id,
      name: $name,
      status: $status,
      duration_ms: $duration_ms,
      failure_count: $failure_count
    }' >>"$gate_results_tmp"
}

run_gate() {
  local gate_id="$1"
  local fn="$2"
  local name replay_cmd gate_dir log_file log_rel
  local started_ms ended_ms duration_ms rc status failure_count
  local summary failure_rel failure_file gate_diag_rel gate_diag_file
  local artifact_paths_json

  name="$(gate_name "$gate_id")"
  replay_cmd="$(gate_replay_cmd "$gate_id")"
  gate_dir="$RUN_DIR/$gate_id"
  mkdir -p "$gate_dir"

  log_file="$gate_dir/${name}.log"
  log_rel="ci-artifacts/$RUN_ID/$gate_id/${name}.log"

  started_ms="$(now_ms)"
  set +e
  "$fn" >"$log_file" 2>&1
  rc=$?
  set -e
  ended_ms="$(now_ms)"
  duration_ms=$((ended_ms - started_ms))

  status="pass"
  failure_count=0
  if [[ $rc -ne 0 ]]; then
    status="fail"
    failure_count=1

    summary="$(trim_summary "$log_file")"
    failure_rel="ci-artifacts/$RUN_ID/$gate_id/failure_diagnostic.v1.json"
    failure_file="$ROOT_DIR/$failure_rel"
    gate_diag_rel="ci-artifacts/$RUN_ID/$gate_id/gate_diagnostic.v1.json"
    gate_diag_file="$ROOT_DIR/$gate_diag_rel"
    artifact_paths_json="$(jq -nc --arg log "$log_rel" --arg diag "$failure_rel" '[ $log, $diag ]')"

    jq -n \
      --arg schema_version "frankenjax.gate-diagnostic.v1" \
      --arg gate_id "$gate_id" \
      --argjson timestamp_unix_ms "$ended_ms" \
      --arg failing_check "$name" \
      --arg error_summary "$summary" \
      --argjson artifact_paths "$artifact_paths_json" \
      --arg replay_cmd "$replay_cmd" \
      --argjson exit_code "$rc" \
      '{
        schema_version: $schema_version,
        gate_id: $gate_id,
        timestamp_unix_ms: $timestamp_unix_ms,
        failing_check: $failing_check,
        error_summary: $error_summary,
        artifact_paths: $artifact_paths,
        replay_cmd: $replay_cmd,
        exit_code: $exit_code
      }' >"$gate_diag_file"

    jq -n \
      --arg schema_version "frankenjax.failure-diagnostic.v1" \
      --arg gate "$gate_id" \
      --arg test "$name" \
      --arg status "fail" \
      --arg summary "$summary" \
      --arg detail_path "$log_rel" \
      --arg replay_cmd "$replay_cmd" \
      --argjson related_fixtures '[]' \
      --argjson timestamp_unix_ms "$ended_ms" \
      --arg error_output "$(tail -c 500 "$log_file" 2>/dev/null || true)" \
      --arg root_cause_hint "dependency_error" \
      '{
        schema_version: $schema_version,
        gate: $gate,
        test: $test,
        status: $status,
        summary: $summary,
        detail_path: $detail_path,
        replay_cmd: $replay_cmd,
        related_fixtures: $related_fixtures,
        timestamp_unix_ms: $timestamp_unix_ms,
        error_output: (if $error_output == "" then null else $error_output end),
        root_cause_hint: $root_cause_hint
      }' >"$failure_file"

    cat "$failure_file" >>"$failure_records_tmp"
    echo "[$gate_id] FAIL (${duration_ms}ms) -> $log_rel" >&2
  else
    echo "[$gate_id] PASS (${duration_ms}ms)"
  fi

  append_gate_result "$gate_id" "$name" "$status" "$duration_ms" "$failure_count"
  return "$rc"
}

pipeline_started_ms="$(now_ms)"
pipeline_blocked=0

for gate_id in G1 G2 G3 G4 G5 G6 G7 G8; do
  name="$(gate_name "$gate_id")"
  if gate_should_skip "$gate_id"; then
    append_gate_result "$gate_id" "$name" "skipped" 0 0
    echo "[$gate_id] SKIPPED (explicit)"
    continue
  fi

  if [[ $pipeline_blocked -eq 1 ]]; then
    append_gate_result "$gate_id" "$name" "skipped" 0 0
    echo "[$gate_id] SKIPPED (blocked by prior gate failure)"
    continue
  fi

  case "$gate_id" in
    G1) run_gate "$gate_id" gate_g1 || pipeline_blocked=1 ;;
    G2) run_gate "$gate_id" gate_g2 || pipeline_blocked=1 ;;
    G3) run_gate "$gate_id" gate_g3 || pipeline_blocked=1 ;;
    G4) run_gate "$gate_id" gate_g4 || pipeline_blocked=1 ;;
    G5) run_gate "$gate_id" gate_g5 || pipeline_blocked=1 ;;
    G6) run_gate "$gate_id" gate_g6 || pipeline_blocked=1 ;;
    G7) run_gate "$gate_id" gate_g7 || pipeline_blocked=1 ;;
    G8) run_gate "$gate_id" gate_g8 || pipeline_blocked=1 ;;
  esac
done

pipeline_finished_ms="$(now_ms)"
pipeline_duration_ms=$((pipeline_finished_ms - pipeline_started_ms))

gate_results_json="$(jq -s '.' "$gate_results_tmp")"
failures_json="$(jq -s '.' "$failure_records_tmp")"

failed_count="$(printf '%s\n' "$gate_results_json" | jq '[.[] | select(.status == "fail")] | length')"
skipped_count="$(printf '%s\n' "$gate_results_json" | jq '[.[] | select(.status == "skipped")] | length')"
passed_count="$(printf '%s\n' "$gate_results_json" | jq '[.[] | select(.status == "pass")] | length')"
overall_passed=true
if [[ "$failed_count" -gt 0 ]]; then
  overall_passed=false
fi

generated_at_ms="$(now_ms)"
jq -n \
  --arg schema_version "frankenjax.reliability-gate-report.v1" \
  --arg run_id "$RUN_ID" \
  --arg budgets_path "$BUDGETS_JSON" \
  --arg artifact_root "$ARTIFACT_ROOT" \
  --argjson generated_at_unix_ms "$generated_at_ms" \
  --argjson full_pipeline_duration_ms "$pipeline_duration_ms" \
  --argjson passed_count "$passed_count" \
  --argjson failed_count "$failed_count" \
  --argjson skipped_count "$skipped_count" \
  --argjson gate_results "$gate_results_json" \
  --argjson failures "$failures_json" \
  --argjson overall_passed "$overall_passed" \
  '{
    schema_version: $schema_version,
    generated_at_unix_ms: $generated_at_unix_ms,
    run_id: $run_id,
    budgets_path: $budgets_path,
    artifact_root: $artifact_root,
    full_pipeline_duration_ms: $full_pipeline_duration_ms,
    gate_summary: {
      passed: $passed_count,
      failed: $failed_count,
      skipped: $skipped_count
    },
    gate_results: $gate_results,
    failures: $failures,
    overall_passed: $overall_passed
  }' >"$GATE_REPORT_JSON"

cp "$GATE_REPORT_JSON" "$RUN_DIR/reliability_gate_report.v1.json"

if [[ -x "$ROOT_DIR/scripts/generate_run_manifest.sh" ]]; then
  "$ROOT_DIR/scripts/generate_run_manifest.sh" \
    --run-id "$RUN_ID" \
    --output "$RUN_DIR" \
    --gate-report "$GATE_REPORT_JSON" >/dev/null || true
fi

echo "Reliability gate report written: $GATE_REPORT_JSON"
echo "Run artifacts: $RUN_DIR"

if [[ "$overall_passed" != "true" ]]; then
  exit 1
fi
