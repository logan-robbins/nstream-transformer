#!/usr/bin/env bash

# Download Wikipedia articles and create raw JSONL shards.
# Requires internet connection.
# Outputs are placed under data/raw/<RUN_ID>.

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
SNAPSHOT=${SNAPSHOT:-20231101.en}
RAW_ROOT=${RAW_ROOT:-data/raw}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-experiments/${RUN_ID}}
LOG_DIR="${EXPERIMENT_DIR}/logs"
RAW_DIR="${RAW_ROOT}/${RUN_ID}"
MANIFEST_PATH=${MANIFEST_PATH:-}
MAX_RECORDS=${MAX_RECORDS:-}
STREAMING=${STREAMING:-1}

mkdir -p "${RAW_DIR}" "${EXPERIMENT_DIR}" "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/ingest_$(date -u +"%Y%m%dT%H%M%SZ").log"
exec > >(tee -a "${RUN_LOG}") 2>&1

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH="src:${PYTHONPATH:-}"

log "INFO | Wikipedia ingest starting"
log "INFO | RUN_ID=${RUN_ID} SNAPSHOT=${SNAPSHOT}"
log "INFO | RAW_DIR=${RAW_DIR}"
log "INFO | PYTHON=$(which python) | $(python --version 2>&1)"

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

build_ingest_cmd() {
  INGEST_CMD=(python scripts/ingest_wikipedia.py --snapshot "${SNAPSHOT}" --output-dir "${RAW_DIR}")
  if [[ "${STREAMING}" == "0" || "${STREAMING}" == "false" ]]; then
    INGEST_CMD+=(--no-streaming)
  else
    INGEST_CMD+=(--streaming)
  fi
  if [[ -n "${MAX_RECORDS}" ]]; then
    INGEST_CMD+=(--max-records "${MAX_RECORDS}")
  fi
  if [[ -n "${MANIFEST_PATH}" ]]; then
    INGEST_CMD+=(--manifest-path "${MANIFEST_PATH}")
  fi
}

build_ingest_cmd
step 1 "Ingest Wikipedia snapshot" "${INGEST_CMD[@]}"

log "INFO | RAW artefacts: ${RAW_DIR}"
find "${RAW_DIR}" -maxdepth 1 -type f -name '*.jsonl' -print | sort || true

cat <<SUMMARY
---
Wikipedia ingest complete.
Run ID: ${RUN_ID}
Raw shards: ${RAW_DIR}
Run log: ${RUN_LOG}

Next step: Run processing with
  RUN_ID=${RUN_ID} bash scripts/run_process.sh
---
SUMMARY

log "INFO | Ingest finished"
