#!/usr/bin/env bash

# Stage-by-stage prep driver turning multistream outputs into KD training JSONL.
# Mirrors logging style from run_process.sh for consistent observability.

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

RUN_ID=${RUN_ID:-wiki-v6-500}
PROCESSED_ROOT=${PROCESSED_ROOT:-data/processed/}
PROCESSED_ROOT=${PROCESSED_ROOT%/}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-experiments/${RUN_ID}}
LOG_DIR="${EXPERIMENT_DIR}/logs"
STAGING_DIR="${EXPERIMENT_DIR}/staging"
MULTISTREAM_PATH="${PROCESSED_ROOT}/${RUN_ID}/multistream.jsonl"
KD_OUTPUT_PATH="${PROCESSED_ROOT}/${RUN_ID}/kd.jsonl"
MAX_RECORDS=${MAX_RECORDS:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ ! -f "${MULTISTREAM_PATH}" ]]; then
  log "ERROR | multistream dataset missing at ${MULTISTREAM_PATH}"
  log "ERROR | Run scripts/run_process.sh to produce multistream first"
  exit 1
fi

mkdir -p "${EXPERIMENT_DIR}" "${LOG_DIR}" "${STAGING_DIR}"

RUN_LOG="${LOG_DIR}/data_prep_$(date -u +"%Y%m%dT%H%M%SZ").log"
exec > >(tee -a "${RUN_LOG}") 2>&1

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH="src:${PYTHONPATH:-}"

log "INFO | data prep starting"
log "INFO | RUN_ID=${RUN_ID}"
log "INFO | multistream=${MULTISTREAM_PATH}"
log "INFO | tokenizer=${TOKENIZER_PATH:-<default>}"

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

build_stage_cmd() {
  local stage="$1"
  STAGE_CMD=(
    "${PYTHON_BIN}" scripts/prepare_kd_dataset.py \
      --stage "${stage}" \
      --multistream "${MULTISTREAM_PATH}" \
      --staging-dir "${STAGING_DIR}" \
      --output "${KD_OUTPUT_PATH}"
  )
  if [[ -n "${MAX_RECORDS}" ]]; then
    STAGE_CMD+=(--max-records "${MAX_RECORDS}")
  fi
  if [[ -n "${TOKENIZER_PATH}" ]]; then
    STAGE_CMD+=(--tokenizer "${TOKENIZER_PATH}")
  fi
  if [[ -n "${MODEL}" ]]; then
    STAGE_CMD+=(--model "${MODEL}")
  fi
}

build_stage_cmd load-multistream
step 1 "Load multistream schema" "${STAGE_CMD[@]}"

build_stage_cmd fan-out-roles
step 2 "Fan out role payloads" "${STAGE_CMD[@]}"

build_stage_cmd tokenize-roles
step 3 "Tokenize role payloads" "${STAGE_CMD[@]}"

build_stage_cmd embed-notes
step 4 "Embed teacher/student notes" "${STAGE_CMD[@]}"

build_stage_cmd pack-kd
step 5 "Pack KD JSONL" "${STAGE_CMD[@]}"

log "INFO | staging artefacts"
find "${STAGING_DIR}" -maxdepth 1 -type f -print | sort || true

cat <<SUMMARY
---
Data prep complete (through stage=pack-kd).
Run ID: ${RUN_ID}
KD output (pending later stages): ${KD_OUTPUT_PATH}
Run log: ${RUN_LOG}
Staging dir: ${STAGING_DIR}
---
SUMMARY

log "INFO | data prep finished"
