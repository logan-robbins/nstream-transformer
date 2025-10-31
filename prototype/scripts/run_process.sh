#!/usr/bin/env bash

# Process raw Wikipedia shards into planner/multistream datasets and validate.
# Runs offline (no internet required) - requires Ollama to be running locally.
# Reads from data/raw/<RUN_ID> and outputs to data/processed/<RUN_ID>.

set -Eeuo pipefail

log() {
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "${ts} | ${1}"
}

on_error() {
  local exit_code=$?
  local line=${1:-unknown}
  log "ERROR | Failed at line ${line}"
  log "ERROR | Command: ${BASH_COMMAND}"
  log "ERROR | Exit code: ${exit_code}"
}
trap 'on_error $LINENO' ERR

if [[ "${TRACE:-1}" == "1" ]]; then
  export PS4='+ $(date +"%H:%M:%S") | ${BASH_SOURCE##*/}:${LINENO} | '
  set -x
fi

RUN_ID=${RUN_ID:-wiki_full}
RAW_ROOT=${RAW_ROOT:-data/raw}
PROCESSED_ROOT=${PROCESSED_ROOT:-data/processed}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-experiments/${RUN_ID}}
LOG_DIR="${EXPERIMENT_DIR}/logs"
RAW_DIR="${RAW_ROOT}/${RUN_ID}"
PROCESSED_DIR="${PROCESSED_ROOT}/${RUN_ID}"
PIPELINE_MAX_RECORDS=${PIPELINE_MAX_RECORDS:-}
INCLUDE_REFERENCES=${INCLUDE_REFERENCES:-0}

# Verify raw data exists
if [[ ! -d "${RAW_DIR}" ]]; then
  log "ERROR | Raw data directory not found: ${RAW_DIR}"
  log "ERROR | Run the ingest script first: RUN_ID=${RUN_ID} bash scripts/run_ingest.sh"
  exit 1
fi

RAW_FILES=$(find "${RAW_DIR}" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
if [[ ${RAW_FILES} -eq 0 ]]; then
  log "ERROR | No JSONL files found in ${RAW_DIR}"
  log "ERROR | Run the ingest script first: RUN_ID=${RUN_ID} bash scripts/run_ingest.sh"
  exit 1
fi

mkdir -p "${PROCESSED_ROOT}" "${EXPERIMENT_DIR}" "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/process_$(date -u +"%Y%m%dT%H%M%SZ").log"
exec > >(tee -a "${RUN_LOG}") 2>&1

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export NSTREAM_DEVICE=${NSTREAM_DEVICE:-mps}
export PYTHONPATH="src:${PYTHONPATH:-}"

log "INFO | Data processing starting"
log "INFO | RUN_ID=${RUN_ID}"
log "INFO | RAW_DIR=${RAW_DIR} PROCESSED_DIR=${PROCESSED_DIR}"
log "INFO | PYTHON=$(which python) | $(python --version 2>&1)"
log "INFO | NSTREAM_DEVICE=${NSTREAM_DEVICE}"

step() {
  local num="$1"; shift
  local title="$1"; shift
  local cmd=("$@")
  local start_ts end_ts dur_s
  start_ts=$(date +%s)
  log "STEP  | [${num}] BEGIN: ${title}"
  log "STEP  | [${num}] CMD: ${cmd[*]}"
  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  end_ts=$(date +%s)
  dur_s=$((end_ts - start_ts))
  log "STEP  | [${num}] END: ${title} (duration=${dur_s}s, rc=${rc})"
  if [[ ${rc} -ne 0 ]]; then
    log "ERROR | Step ${num} failed"
    exit ${rc}
  fi
}

build_pipeline_cmd() {
  PIPE_CMD=(python scripts/prepare_wiki_multistream.py --raw-dir "${RAW_DIR}" --output-dir "${PROCESSED_ROOT}" --run-id "${RUN_ID}")
  if [[ -n "${PIPELINE_MAX_RECORDS}" ]]; then
    PIPE_CMD+=(--max-records "${PIPELINE_MAX_RECORDS}")
  fi
  if [[ "${INCLUDE_REFERENCES}" == "1" || "${INCLUDE_REFERENCES}" == "true" ]]; then
    PIPE_CMD+=(--include-references)
  fi
}

build_validation_cmd() {
  VALIDATE_CMD=(python scripts/validate_wiki_schema.py --input "${PROCESSED_DIR}/multistream.jsonl" --output "${PROCESSED_DIR}/schema_report.json")
}

log "INFO | RAW artefacts: ${RAW_DIR}"
find "${RAW_DIR}" -maxdepth 1 -type f -name '*.jsonl' -print | sort || true

build_pipeline_cmd
step 1 "Transform raw shards into planner/multistream" "${PIPE_CMD[@]}"

log "INFO | Processed artefacts: ${PROCESSED_DIR}"
ls -lh "${PROCESSED_DIR}" || true

build_validation_cmd
step 2 "Validate multistream schema" "${VALIDATE_CMD[@]}"

log "INFO | Schema report preview"
head -n 10 "${PROCESSED_DIR}/schema_report.json" || true

cat <<SUMMARY
---
Data processing complete.
Run ID: ${RUN_ID}
Raw shards: ${RAW_DIR}
Planner dataset: ${PROCESSED_DIR}/planner.jsonl
Multistream dataset: ${PROCESSED_DIR}/multistream.jsonl
Schema report: ${PROCESSED_DIR}/schema_report.json
Stats: ${PROCESSED_DIR}/stats.json
Run log: ${RUN_LOG}
---
SUMMARY

log "INFO | Data processing finished"
