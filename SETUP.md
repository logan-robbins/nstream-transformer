# nstream-transformer
Reference implementation of the N‑Stream Transformer: parallel multi‑role decoding on a single shared decoder, with Dynamic Notes Bus (DNB) and Shared Notes Cross‑Attention (SNC) for causal inter‑lane communication. Includes a paper‑aligned inference runtime and a lean PEFT‑ready training loop over GPT‑OSS trunks.

## Lambda Cloud GPU setup

The project is configured to use Python 3.12 and PyTorch GPU wheels without system‑level changes. On Lambda Cloud, run:

```
prototype/scripts/setup_lambda_gpu.sh
```

The script will:
- Detect your CUDA driver version with `nvidia-smi`.
- Select the closest matching official PyTorch CUDA wheel index (e.g., `cu124`).
- Install Python 3.12 via `uv` and sync the environment.
- Verify that PyTorch sees the GPU.

If you prefer to run steps manually:

```
uv python install 3.12
export UV_INDEX_URL=https://download.pytorch.org/whl/cu124  # for CUDA 12.4+ drivers
uv sync --python 3.12 -E gpu
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

No `xformers` package is required by this codebase; GPU execution relies only on PyTorch.
