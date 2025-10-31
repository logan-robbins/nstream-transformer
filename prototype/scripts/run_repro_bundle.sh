#!/usr/bin/env bash

# One-stop smoke pipeline: data transform → KD prep → train → evaluate → infer.
# Intended for local reproducibility checks with fixed seeds and small configs.

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

if [[ "${TRACE:-0}" == "1" ]]; then
  export PS4='+ $(date +"%H:%M:%S") | ${BASH_SOURCE##*/}:${LINENO} | '
  set -x
fi

RUN_ID=${RUN_ID:-wiki_smoke}
CONFIG=${CONFIG:-configs/wiki_smoke.yaml}
SEED=${SEED:-1234}
PROMPT=${PROMPT:-"Summarise the latest wiki section into three bullet notes."}
PIPELINE_MAX_RECORDS=${PIPELINE_MAX_RECORDS:-32}
KD_MAX_RECORDS=${KD_MAX_RECORDS:-32}
RAW_ROOT=${RAW_ROOT:-data/raw}
PROCESSED_ROOT=${PROCESSED_ROOT:-data/processed}
NSTREAM_DEVICE=${NSTREAM_DEVICE:-cpu}

RAW_DIR="${RAW_ROOT}/${RUN_ID}"
PROCESSED_DIR="${PROCESSED_ROOT%/}/${RUN_ID}"
EXPERIMENT_DIR=${EXPERIMENT_DIR:-experiments/${RUN_ID}}
LOG_DIR="${EXPERIMENT_DIR}/logs"
EVAL_DIR="${EXPERIMENT_DIR}/evaluation"
INFER_MANIFEST="${EXPERIMENT_DIR}/infer/manifest.json"

mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH="src:${PYTHONPATH:-}"
export NSTREAM_DEVICE
export PYTHONHASHSEED=${SEED}

log "INFO | Repro bundle starting"
log "INFO | RUN_ID=${RUN_ID} CONFIG=${CONFIG} SEED=${SEED}"
log "INFO | NSTREAM_DEVICE=${NSTREAM_DEVICE}"
log "INFO | RAW_DIR=${RAW_DIR}"
log "INFO | PROCESSED_DIR=${PROCESSED_DIR}"
log "INFO | EXPERIMENT_DIR=${EXPERIMENT_DIR}"

if [[ ! -d "${RAW_DIR}" ]]; then
  log "ERROR | Raw data missing at ${RAW_DIR}"
  log "ERROR | Run the ingest helper first: RUN_ID=${RUN_ID} bash scripts/run_ingest.sh"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  log "ERROR | python interpreter not found on PATH"
  exit 1
fi

log "INFO | Python=$(which python) | $(python --version 2>&1)"

step() {
  local idx="$1"; shift
  local title="$1"; shift
  local cmd=("$@")
  local start_ts end_ts dur_s
  start_ts=$(date +%s)
  log "STEP  | [${idx}] BEGIN: ${title}"
  log "STEP  | [${idx}] CMD: ${cmd[*]}"
  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  end_ts=$(date +%s)
  dur_s=$((end_ts - start_ts))
  log "STEP  | [${idx}] END: ${title} (duration=${dur_s}s, rc=${rc})"
  if [[ ${rc} -ne 0 ]]; then
    log "ERROR | Step ${idx} failed"
    exit ${rc}
  fi
}

PIPELINE_ENV=(
  RUN_ID="${RUN_ID}"
  PROCESSED_ROOT="${PROCESSED_ROOT}"
  PIPELINE_MAX_RECORDS="${PIPELINE_MAX_RECORDS}"
  TRACE="${TRACE:-0}"
)

step 1 "Transform raw shards → planner/multistream" env "${PIPELINE_ENV[@]}" bash scripts/run_process.sh

DATA_PREP_ENV=(
  RUN_ID="${RUN_ID}"
  PROCESSED_ROOT="${PROCESSED_ROOT}"
  MAX_RECORDS="${KD_MAX_RECORDS}"
  TRACE="${TRACE:-0}"
)

step 2 "Pack KD JSONL" env "${DATA_PREP_ENV[@]}" bash scripts/run_data_prep.sh

TRAIN_CMD=(python scripts/train.py --config "${CONFIG}")
step 3 "Train adapters (seeded)" "${TRAIN_CMD[@]}"

ADAPTER_CKPT="${EXPERIMENT_DIR}/adapters.pt"
if [[ ! -f "${ADAPTER_CKPT}" ]]; then
  log "ERROR | Expected adapters checkpoint missing at ${ADAPTER_CKPT}"
  exit 1
fi

mkdir -p "${EVAL_DIR}"
EVAL_CMD=(
  python scripts/evaluate.py
  --config "${CONFIG}"
  --checkpoint "${ADAPTER_CKPT}"
  --output-dir "${EVAL_DIR}"
)
step 4 "Evaluate adapters" "${EVAL_CMD[@]}"

mkdir -p "$(dirname "${INFER_MANIFEST}")"
INFER_CMD=(
  python scripts/infer.py
  --config "${CONFIG}"
  --checkpoint "${ADAPTER_CKPT}"
  --prompt "${PROMPT}"
  --seed "${SEED}"
  --output "${INFER_MANIFEST}"
  --max-new-tokens 32
)
step 5 "Run deterministic inference" "${INFER_CMD[@]}"

EVAL_MANIFEST="${EVAL_DIR}/evaluation_manifest.json"
KD_JSONL="${PROCESSED_DIR}/kd.jsonl"
TRAIN_MANIFEST="${EXPERIMENT_DIR}/train_manifest.json"

cat <<SUMMARY
---
Repro bundle complete.
  KD dataset ............. ${KD_JSONL}
  Adapter checkpoint ..... ${ADAPTER_CKPT}
  Training manifest ...... ${TRAIN_MANIFEST}
  Evaluation manifest .... ${EVAL_MANIFEST}
  Inference manifest ..... ${INFER_MANIFEST}
Logs live under ${LOG_DIR}.
Rerun with different seeds via: SEED=<value> bash scripts/run_repro_bundle.sh
---
SUMMARY

log "INFO | Repro bundle finished"
