#!/usr/bin/env bash
set -euo pipefail

# Minimal GPU setup for Lambda Cloud machines without system changes.
# - Detect CUDA driver version
# - Choose PyTorch CUDA wheel index
# - Install Python 3.12 via uv
# - Sync deps and verify GPU

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is required. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

CUDA_INDEX=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER_RAW=$(nvidia-smi | awk '/CUDA Version:/ {print $NF; exit}') || true
  # Expect something like 12.8
  if [[ -n "${CUDA_VER_RAW:-}" ]]; then
    CUDA_MAJOR=${CUDA_VER_RAW%%.*}
    CUDA_MINOR=${CUDA_VER_RAW##*.}
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
      # Map any CUDA 12.x to cu124 wheels (broadly compatible on newer drivers)
      CUDA_INDEX="https://download.pytorch.org/whl/cu124"
    elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
      CUDA_INDEX="https://download.pytorch.org/whl/cu118"
    fi
  fi
fi

echo "Detected CUDA driver: ${CUDA_VER_RAW:-none}"
if [[ -n "$CUDA_INDEX" ]]; then
  echo "Using PyTorch index: $CUDA_INDEX"
else
  echo "No matching CUDA index detected; using CPU wheels from PyPI."
fi

echo "Installing Python 3.12 via uv..."
uv python install 3.12

echo "Syncing project dependencies..."
if [[ -n "$CUDA_INDEX" ]]; then
  UV_INDEX_URL="$CUDA_INDEX" uv sync --python 3.12 -E gpu
else
  uv sync --python 3.12
fi

echo "Verifying PyTorch GPU availability..."
./.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", getattr(torch.version, "cuda", None))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY

echo "Setup complete."

