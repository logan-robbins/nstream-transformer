# N‑Stream Transformer — Parallel Inference (Current State)

This repository provides a paper‑aligned N‑Stream inference runtime (Dynamic Notes Bus + Shared Notes Cross‑Attention) over a frozen GPT‑OSS trunk. The current path is a minimal, training‑free POC intended for a technical audience.

What runs today

- Frozen GPT‑OSS trunk (shared decoder + LM head).
- Per‑role adapters and SNC attending to a lagged, versioned Dynamic Notes Bus (Δ, K, triangular topology).
- Agreement‑gated rollbacks of the last L tokens and stride‑based scheduling across roles.
- Plan coverage diagnostics; no trained planner/seeder yet.

Bus semantics (embeddings‑only)

- The DNB carries fixed‑dim embedding vectors (`notes_dim`) and SNC consumes them as K/V for cross‑attention. We do not pass text across lanes.
- Any “plan” or “notes” tokens we surface are for observability/coverage only; when present they are embedded inside the model and the embeddings are what enter the bus.

How we demonstrate divergence (no training)

- Pre‑write a very short plan (≤30 tokens per role) and inject it into the Notes Bus at t0. SNC conditions each lane on these seeds immediately, showing parallelism and causal influence without fine‑tuning.

Install (minimal)

```
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install transformers peft accelerate pyyaml tiktoken protobuf
# Choose one: CPU wheel or CUDA wheel per your host
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# or: pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -e .
```

Prepare roles and seeds

role_prefixes.json

```
{
  "stream_1": "You are Part 1. Focus only on part 1: ",
  "stream_2": "You are Part 2. Focus only on part 2: ",
  "stream_3": "You are Part 3. Focus only on part 3: "
}
```

seed_texts.json (≤30 tokens per role)

```
{
  "stream_1": "Plan: early US history, colonization, independence, constitution.",
  "stream_2": "Plan: modern demographics, economy, federal government structure.",
  "stream_3": "Plan: culture, science, global alliances of the United States."
}
```

Run inference

```
python scripts/infer.py --config configs/gpt_oss_transfer.yaml \
  --prompt "Tell me some facts about the US." \
  --role stream_1 --role stream_2 --role stream_3 \
  --role-prefix-file role_prefixes.json \
  --seed-text-file seed_texts.json \
  --read-lag-delta 0 --alpha 1 --gate-g 1 --max-new-tokens 512 --verbose
```

Flags that matter

- `--role` to set lane names; `--role-prefix-file` for per‑role prompt prefixes.
- `--seed-text-file` (preferred) or `--seed-text role=text` to inject t0 plan seeds into the DNB.
- `--read-lag-delta` (Δ) controls snapshot reveal; use 0 for immediate visibility.
- `--alpha` blends attended vs base logits (1 = attended only).
- `--gate-g` scales SNC residual (0 disables cross‑lane influence; 1 maximizes it).
- `--max-new-tokens` (default 512 if not set) limits tokens per role.
- `--stream-jsonl` emits per‑token JSON; a manifest is always written to `experiments/infer/manifest.json`.

Local weights layout (GPT‑OSS‑20B)

- The default config (`configs/gpt_oss_transfer.yaml`) points `model.trunk.base_model` to `gpt-oss-20b/original`. The CLI resolves the tokenizer at `gpt-oss-20b/tokenizer` next to it. Place these folders under the `prototype/` directory (where the CLI scripts live) so relative paths resolve:

```
prototype/
  gpt-oss-20b/
    original/
      config.json
      generation_config.json
      model.safetensors.index.json
      model-00001-of-00003.safetensors
      model-00002-of-00003.safetensors
      model-00003-of-00003.safetensors
    tokenizer/
      tokenizer.json
      tokenizer.model
      added_tokens.json
      special_tokens_map.json
      tokenizer_config.json
      chat_template.jinja
```

- If you store weights elsewhere, either:
  - edit `model.trunk.base_model` in your YAML to an absolute path, or
  - pass a different config via `--config ...` that points to your path.

- The loader runs fully offline when given a local path. If you instead pass a remote model id, ensure your environment has network access and the required model license is accepted.

Future direction (fine‑tuning)

- We aim to fine‑tune planner/speculation heads so the model emits plan/notes tokens that are embedded and pushed into the DNB automatically, removing the manual seed step. The training approach follows FULL.md (asynchronous reveal; KD + stability) and keeps teacher stop‑grad/EMA semantics.
