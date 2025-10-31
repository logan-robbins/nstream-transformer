# AGENTS.md — Working Guide for AI Coding Agents

This repository implements the N‑Stream Transformer as specified in `docs/drafts/v7/FULL.md` and the inference/runtime spec in `docs/drafts/v7/INFERENCE.md`. Use this file as the authoritative playbook when editing code.

## Ground Truth Specs
- Follow the paper‑aligned specs strictly:
  - Inference/runtime: `docs/drafts/v7/INFERENCE.md`
  - Training: `docs/TRAINING.md`
  - Full text: `docs/drafts/v7/FULL.md`
- Do not scaffold or add “optional/legacy” modes. Teacher is mandatory in training.

## Project Map (files you’ll touch most)
- Models: `src/nstream_transformer/models/`
  - `nstream_transformer.py` — role adapters, SNC cross‑attn, heads; planner logits from SNC‑conditioned states.
  - `role_adapters.py`, `heads/*` — keep APIs stable; adapter‑only checkpoints rely on these names.
- Inference: `src/nstream_transformer/inference/`
  - `orchestrator.py` (decode loop), `window.py` (Δ/topology windows), `dnb_bus.py` (K/Δ/storage), `scheduler.py` (stride B), `config.py` (mapping from training → runtime).
- Training: `src/nstream_transformer/training/trainer.py`
  - Curriculum (B/L/Δ, stages), exposure dials, KD/stability/structure losses, micro‑rollouts, GradNorm.
- Data: `src/nstream_transformer/data/collator_kd.py` (two‑branch bus packing), `teacher_provider.py` (runner + caching), `tokenizer.py`.
- CLIs: `scripts/train.py`, `scripts/evaluate.py`, `scripts/infer.py` (adapter checkpoint load, manifest write).

## Hard Invariants (preserve these)
- Planner logits are computed from SNC‑conditioned (attended) states during training.
- Training requires `training.teacher_runner`; fail fast if missing.
- DNB invariants: retain ≤ K snapshots, per‑producer monotonic reads, Δ‑lag enforced, triangular topology by default.
- Curriculum behavior: stride B releases commit mask, horizon L bounds rollback/stability, Δ applied consistently to teacher/student and Δ+1 to `pre_notes`.
- Adapter checkpoints: save only adapters/heads (`adapters.pt`); trunk weights come from HF.

## Typical Tasks and How To Do Them
- Add a new diagnostic or loss:
  - Gate it by stage in `Trainer._active_loss_weights`; compute in `_compute_losses` with proper masks; log under `self.metric_history`.
  - Add a focused unit test near the module and, if needed, an integration test.
- Change inference behavior (e.g., cadence/gating):
  - Adjust orchestrator + config; update tests: `tests/unit/test_notes_window_builder.py`, `tests/unit/test_scheduler.py`, `tests/integration/test_orchestrator_runtime.py`.
- Touch bus/window logic:
  - Keep Δ‑lag, version monotonicity, and K compaction. Extend `NotesWindowBuilder` and update unit tests accordingly.
- Add CLI flags:
  - Parse in the script, thread through the relevant config, and document in README.

## Testing & Validation
- Run fast tests locally: `PYTHONPATH=src python -m pytest --maxfail=1`.
- Coverage targets the moving pieces:
  - Δ/topology windows, stride scheduler, rollback/KV state, orchestrator integration, curriculum with teacher runner.
- Keep dependencies optional in tests; do not require network. Tests use stubs/fakes and guards around heavy imports.

## Style & Conventions
- Python: 4 spaces, `snake_case` for functions/modules, `PascalCase` for dataclasses.
- Keep public APIs stable for CLIs and model entry points; add flags conservatively.
- Log via `nstream_transformer.utils.configure_logging`.

## Adapter Checkpoints
- Train writes: `training.telemetry_dir/adapters.pt` (see `scripts/train.py`).
- Evaluate/Infer loads: `--checkpoint <file|dir>`; `scripts/evaluate.py` and `scripts/infer.py` call `model.load_adapters(state, strict=False)`.

## Gotchas
- Don’t remove Δ+1 when building `pre_notes` for stability.
- Notes/speculative MSE must match `[batch, roles, dim]`; pool student predictions over time when needed (already implemented).
- Keep teacher stop‑grad/EMA semantics; do not backprop through teacher.
- Avoid adding network calls in code paths exercised by tests.

## Commit Discipline
- Use `scope: imperative summary` (e.g., `inference: anneal SNC gate on low agreement`). Keep ≤ 72 chars; explain behavior change in body.
- Every config/loss/flag addition must ship a regression test and a doc note (README or TRAINING.md).

## Build & Run Cheatsheet
- Env: `python -m venv .venv && source .venv/bin/activate && python -m pip install -e .[test]`.
- Train: `python scripts/train.py --config configs/gpt_oss_transfer.yaml`.
- Eval: `python scripts/evaluate.py --config configs/gpt_oss_transfer.yaml --checkpoint experiments/gpt_oss/adapters.pt --eval-dataset data/eval.jsonl`.
- Infer: `python scripts/infer.py --config configs/gpt_oss_transfer.yaml --checkpoint experiments/gpt_oss/adapters.pt --prompt "..." --seed 42`.

## When In Doubt
- Re‑read `docs/TRAINING.md` and `docs/drafts/v7/INFERENCE.md`.
- Mirror any behavior changes with tests and a brief doc update.
- Prefer minimal deltas that keep invariants and existing APIs intact.
