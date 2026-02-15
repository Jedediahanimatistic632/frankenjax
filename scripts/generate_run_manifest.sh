#!/usr/bin/env bash
# generate_run_manifest.sh — Produce a CI run manifest with failure diagnostics
# and artifact index for the current run.
#
# Usage:
#   ./scripts/generate_run_manifest.sh [--run-id <id>] [--output <path>] [--gate-report <path>]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
OUTPUT_DIR=""
GATE_REPORT="$ROOT_DIR/artifacts/ci/reliability_gate_report.v1.json"

usage() {
  echo "Usage: $0 [--run-id <id>] [--output <dir>] [--gate-report <path>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --gate-report) GATE_REPORT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  git_sha="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  RUN_ID="${timestamp}-${git_sha}"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/artifacts/ci/runs/$RUN_ID"
fi

mkdir -p "$OUTPUT_DIR"

STARTED_AT="$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')"
RUST_VERSION="$(rustc --version 2>/dev/null || echo "unknown")"
OS_INFO="$(uname -srm 2>/dev/null || echo "unknown")"
GIT_SHA_FULL="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")"
GIT_BRANCH="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"

CI_RUN_DIR="$ROOT_DIR/ci-artifacts/$RUN_ID"

declare -a ARTIFACT_INDEX=()
ARTIFACT_COUNT=0

add_artifact_entry() {
  local file="$1"
  local category="$2"
  local rel_path size sha gate_id test_id entry

  rel_path="${file#$ROOT_DIR/}"
  size="$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)"
  sha="$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || echo "")"
  gate_id=""
  test_id=""

  if [[ "$category" == "failure_diagnostic" ]]; then
    gate_id="$(jq -r '.gate // empty' "$file" 2>/dev/null || true)"
    test_id="$(jq -r '.test // empty' "$file" 2>/dev/null || true)"
  fi

  entry="$(jq -nc \
    --arg path "$rel_path" \
    --arg category "$category" \
    --argjson size_bytes "$size" \
    --arg sha256 "$sha" \
    --arg gate_id "$gate_id" \
    --arg test_id "$test_id" \
    '{
      path: $path,
      category: $category,
      size_bytes: $size_bytes,
      sha256: $sha256
    }
    + (if $gate_id != "" then {gate_id: $gate_id} else {} end)
    + (if $test_id != "" then {test_id: $test_id} else {} end)')"

  ARTIFACT_INDEX+=("$entry")
  ARTIFACT_COUNT=$((ARTIFACT_COUNT + 1))
}

scan_artifacts() {
  local dir="$1"
  local category="$2"
  local pattern="$3"

  if [[ ! -d "$dir" ]]; then
    return
  fi

  while IFS= read -r -d '' file; do
    add_artifact_entry "$file" "$category"
  done < <(find "$dir" -name "$pattern" -type f -print0 2>/dev/null)
}

scan_artifacts "$ROOT_DIR/artifacts/testing/logs" "test_log" "*.json"
scan_artifacts "$ROOT_DIR/artifacts/e2e" "e2e_log" "*.e2e.json"
scan_artifacts "$ROOT_DIR/artifacts/e2e/golden_journeys" "golden_journey" "*.golden.json"
scan_artifacts "$ROOT_DIR/artifacts/testing" "coverage_report" "*.coverage.*.json"
scan_artifacts "$ROOT_DIR/artifacts/ci" "other" "*.v1.json"
scan_artifacts "$ROOT_DIR/artifacts/performance/evidence" "perf_delta" "*.json"
scan_artifacts "$ROOT_DIR/artifacts/crash-triage" "crash_triage" "*.json"
scan_artifacts "$ROOT_DIR/artifacts/durability" "durability_sidecar" "*.json"

if [[ -d "$CI_RUN_DIR" ]]; then
  scan_artifacts "$CI_RUN_DIR" "failure_diagnostic" "failure_diagnostic.v1.json"
  scan_artifacts "$CI_RUN_DIR" "other" "gate_diagnostic.v1.json"
  scan_artifacts "$CI_RUN_DIR" "other" "*.log"
fi

if [[ "$OUTPUT_DIR" != "$CI_RUN_DIR" && -d "$OUTPUT_DIR" ]]; then
  scan_artifacts "$OUTPUT_DIR" "failure_diagnostic" "failure_diagnostic.v1.json"
  scan_artifacts "$OUTPUT_DIR" "other" "gate_diagnostic.v1.json"
  scan_artifacts "$OUTPUT_DIR" "other" "*.log"
fi

GATE_RESULTS_JSON='[]'
FAILURES_JSON='[]'
OVERALL="pass"

if [[ -f "$GATE_REPORT" ]]; then
  if jq -e '.gate_results | type == "array"' "$GATE_REPORT" >/dev/null 2>&1; then
    GATE_RESULTS_JSON="$(jq -c '.gate_results' "$GATE_REPORT")"
    OVERALL="$(jq -r 'if (.overall_passed // .overall_pass // true) then "pass" else "fail" end' "$GATE_REPORT")"
    if jq -e '.failures | type == "array"' "$GATE_REPORT" >/dev/null 2>&1; then
      FAILURES_JSON="$(jq -c '.failures' "$GATE_REPORT")"
    fi
  else
    GATE_RESULTS_JSON="$(jq -c '
      [
        (if has("coverage") then {gate_id:"coverage",name:"coverage",status:(if .coverage.passed then "pass" else "fail" end),duration_ms:0} else empty end),
        (if has("flake") then {gate_id:"flake",name:"flake",status:(if .flake.passed then "pass" else "fail" end),duration_ms:0} else empty end),
        (if has("runtime") then {gate_id:"runtime",name:"runtime",status:(if .runtime.passed then "pass" else "fail" end),duration_ms:0} else empty end),
        (if has("crash") then {gate_id:"crash",name:"crash",status:(if .crash.passed then "pass" else "fail" end),duration_ms:0} else empty end),
        (if has("perf") then {gate_id:"perf",name:"perf",status:(if .perf.passed then "pass" else "fail" end),duration_ms:0} else empty end)
      ]' "$GATE_REPORT")"
    OVERALL="$(jq -r 'if (.overall_passed // .overall_pass // true) then "pass" else "fail" end' "$GATE_REPORT")"
  fi
fi

mapfile -t failure_files < <(
  {
    [[ -d "$CI_RUN_DIR" ]] && find "$CI_RUN_DIR" -name 'failure_diagnostic.v1.json' -type f
    [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -name 'failure_diagnostic.v1.json' -type f
  } 2>/dev/null | sort -u
)

if (( ${#failure_files[@]} > 0 )); then
  FAILURES_JSON="$(jq -s '.' "${failure_files[@]}")"
fi

TOTAL_TESTS="$(printf '%s\n' "$GATE_RESULTS_JSON" | jq 'length')"
PASSED="$(printf '%s\n' "$GATE_RESULTS_JSON" | jq '[.[] | select(.status == "pass")] | length')"
FAILED="$(printf '%s\n' "$GATE_RESULTS_JSON" | jq '[.[] | select(.status == "fail")] | length')"
SKIPPED="$(printf '%s\n' "$GATE_RESULTS_JSON" | jq '[.[] | select(.status == "skipped")] | length')"

if [[ "$TOTAL_TESTS" -eq 0 ]]; then
  FAILED="$(printf '%s\n' "$FAILURES_JSON" | jq 'length')"
  TOTAL_TESTS="$FAILED"
  PASSED=0
  SKIPPED=0
fi

if [[ "$FAILED" -gt 0 ]]; then
  OVERALL="fail"
fi

FINISHED_AT="$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')"
DURATION=$((FINISHED_AT - STARTED_AT))

ARTIFACT_INDEX_JSON='[]'
if (( ${#ARTIFACT_INDEX[@]} > 0 )); then
  ARTIFACT_INDEX_JSON="$(printf '%s\n' "${ARTIFACT_INDEX[@]}" | jq -s '.')"
fi

jq -n \
  --arg schema_version "frankenjax.run-manifest.v1" \
  --arg run_id "$RUN_ID" \
  --argjson started_at_unix_ms "$STARTED_AT" \
  --argjson finished_at_unix_ms "$FINISHED_AT" \
  --argjson total_duration_ms "$DURATION" \
  --argjson total_tests "$TOTAL_TESTS" \
  --argjson passed "$PASSED" \
  --argjson failed "$FAILED" \
  --argjson skipped "$SKIPPED" \
  --arg overall_status "$OVERALL" \
  --argjson gate_results "$GATE_RESULTS_JSON" \
  --argjson failures "$FAILURES_JSON" \
  --argjson artifact_index "$ARTIFACT_INDEX_JSON" \
  --arg rust_version "$RUST_VERSION" \
  --arg os "$OS_INFO" \
  --arg git_sha "$GIT_SHA_FULL" \
  --arg git_branch "$GIT_BRANCH" \
  '{
    schema_version: $schema_version,
    run_id: $run_id,
    started_at_unix_ms: $started_at_unix_ms,
    finished_at_unix_ms: $finished_at_unix_ms,
    total_duration_ms: $total_duration_ms,
    summary: {
      total_tests: $total_tests,
      passed: $passed,
      failed: $failed,
      skipped: $skipped,
      flaky: 0,
      overall_status: $overall_status
    },
    gate_results: $gate_results,
    failures: $failures,
    artifact_index: $artifact_index,
    env: {
      rust_version: $rust_version,
      os: $os,
      git_sha: $git_sha,
      git_branch: $git_branch
    }
  }' >"$OUTPUT_DIR/manifest.json"

echo "[manifest] wrote $OUTPUT_DIR/manifest.json ($ARTIFACT_COUNT artifacts indexed)"

cat >"$OUTPUT_DIR/summary.txt" <<SUMMARY
============================================================
  FrankenJAX CI Run Summary
  Run ID:   $RUN_ID
  Branch:   $GIT_BRANCH
  Commit:   $GIT_SHA_FULL
  Date:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')
============================================================

RESULT: $(echo "$OVERALL" | tr '[:lower:]' '[:upper:]')

Tests:    $TOTAL_TESTS total, $PASSED passed, $FAILED failed, $SKIPPED skipped
Artifacts: $ARTIFACT_COUNT indexed

SUMMARY

if [[ "$(printf '%s\n' "$GATE_RESULTS_JSON" | jq 'length')" -gt 0 ]]; then
  echo "Gate Results:" >>"$OUTPUT_DIR/summary.txt"
  printf '%s\n' "$GATE_RESULTS_JSON" | jq -r '.[] | "  \(.gate_id)\t\(.status)"' >>"$OUTPUT_DIR/summary.txt"
  echo "" >>"$OUTPUT_DIR/summary.txt"
fi

if [[ "$FAILED" -gt 0 ]]; then
  echo "FAILURES:" >>"$OUTPUT_DIR/summary.txt"
  echo "------------------------------------------------------------" >>"$OUTPUT_DIR/summary.txt"
  printf '%s\n' "$FAILURES_JSON" | jq -r '.[] | "  TEST:   \(.test)\n  REASON: \(.summary)\n  REPLAY: \(.replay_cmd)\n"' >>"$OUTPUT_DIR/summary.txt"
else
  echo "No failures detected." >>"$OUTPUT_DIR/summary.txt"
fi

echo "------------------------------------------------------------" >>"$OUTPUT_DIR/summary.txt"
echo "Manifest: $OUTPUT_DIR/manifest.json" >>"$OUTPUT_DIR/summary.txt"
echo "Artifacts dir: $ROOT_DIR/artifacts/" >>"$OUTPUT_DIR/summary.txt"

echo "[manifest] wrote $OUTPUT_DIR/summary.txt"
cat "$OUTPUT_DIR/summary.txt"
