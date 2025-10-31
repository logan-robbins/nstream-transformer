"""Quick benchmark loader for raw GPT-OSS models.

Loads a local GPT-OSS directory and times tokenizer/model load and a short
generation on a single prompt. Keeps dependencies minimal and does not touch
the N-Stream runtime. Designed for local, offline weights.

Usage examples:

  PYTHONPATH=src python scripts/bench_gpt_oss.py \
    --model-dir gpt-oss-20b/original --tokenizer-dir gpt-oss-20b/tokenizer \
    --prompt "Hello, my name is" --max-new-tokens 32 --device auto

"""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os, sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

try:
    import torch
except Exception as _exc:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_ERROR = _exc
else:
    _TORCH_ERROR = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as _exc:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _TXF_ERROR = _exc
else:
    _TXF_ERROR = None

try:
    # Optional: use project logger if available
    from nstream_transformer.utils import configure_logging as _nstream_configure_logging
except Exception:
    _nstream_configure_logging = None  # type: ignore


def configure_logging() -> logging.Logger:
    if _nstream_configure_logging is not None:
        return _nstream_configure_logging(logging.INFO, name="nstream.cli.bench_gpt_oss")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger("bench_gpt_oss")


def pick_device(preferred: str | None = None) -> str:
    """Best-effort device resolution with CUDA preference.

    Accepts explicit choices (cpu/cuda/mps). When preferred is "auto" or None,
    prefers CUDA, then MPS, then CPU. This mirrors the project-wide resolver.
    """
    if preferred in {"cpu", "cuda", "mps"}:
        return preferred
    # treat anything else (including None/"auto") as auto-detect
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, alias: Optional[str]) -> str:
    if alias:
        return alias
    # Sensible defaults per device
    if device == "cuda":
        return "bfloat16"
    if device == "mps":
        return "float16"  # mps + bf16 is not supported reliably
    return "float32"


def resolve_tokenizer_dir(model_dir: Path, explicit_tokenizer: Optional[Path]) -> Optional[Path]:
    if explicit_tokenizer is not None:
        return explicit_tokenizer
    # Try sibling directory named "tokenizer" at repo-level or under model dir
    # 1) model_dir/../tokenizer (repo layout: gpt-oss-20b/original + gpt-oss-20b/tokenizer)
    cand1 = model_dir.parent / "tokenizer"
    if cand1.is_dir():
        return cand1
    # 2) model_dir/tokenizer
    cand2 = model_dir / "tokenizer"
    if cand2.is_dir():
        return cand2
    # 3) model_dir itself typically contains tokenizer.json; HF can use it
    return model_dir if (model_dir / "tokenizer.json").exists() else None


def main() -> None:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required but not available: %s" % (repr(_TXF_ERROR),)
        )
    if torch is None:
        raise RuntimeError("torch is required but not available: %s" % (repr(_TORCH_ERROR),))

    import platform

    # Detect OS and best-available device to use as default
    os_name = platform.system()
    default_device = pick_device(None)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("gpt-oss-20b/original"), help="Path to local model directory.")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="Path to tokenizer directory (optional).")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help=(
            "Runtime device preference (auto selects CUDA if available, then MPS, else CPU)."
        ),
    )
    parser.add_argument("--torch-dtype", dest="torch_dtype", default=None, help="Torch dtype alias (e.g. bfloat16, float16, float32).")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens to generate.")
    parser.add_argument("--min-new-tokens", type=int, default=None, help="Optional minimum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (ignored if --greedy).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p (ignored if --greedy).")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (ignored if --greedy).")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width (1 disables beam search).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional repetition penalty.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None, help="Optional no-repeat ngram size.")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling and use greedy decoding.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    # GPT-OSS specific useful knobs
    parser.add_argument("--output-router-logits", action="store_true", help="Return MoE router logits in outputs.")
    parser.add_argument("--logits-to-keep", type=int, default=0, help="Keep only top-K logits on return (0 to disable).")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", dest="use_cache", action="store_true", help="Force KV cache usage.")
    cache_group.add_argument("--no-use-cache", dest="no_use_cache", action="store_true", help="Disable KV cache.")
    # Prompt shaping for "thinking" and system directives
    parser.add_argument("--think-prefix", type=str, default=None, help="Optional prefix to prepend to the prompt (e.g., <think>).")
    parser.add_argument("--think-suffix", type=str, default=None, help="Optional suffix to append to the prompt (e.g., </think>).")
    parser.add_argument(
        "--reasoning-level",
        choices=["low", "medium", "high", "deep"],
        default=None,
        help="Inject a system hint like 'System: Reasoning: <level>' before the prompt.",
    )
    parser.add_argument(
        "--system-directive",
        type=str,
        default=None,
        help="Arbitrary system directive to prepend (e.g., 'Be concise and show steps.').",
    )
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Prompt text.")
    parser.add_argument("--print-output", action="store_true", help="Print generated text to stdout.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional path to write a JSON summary of timings.")
    args = parser.parse_args()

    logger = configure_logging()
    logger.info("environment | os=%s | default_device=%s | torch_cuda=%s | torch_mps=%s",
                os_name,
                default_device,
                getattr(torch, 'cuda', None) and torch.cuda.is_available(),
                getattr(getattr(torch, 'backends', None), 'mps', None) and torch.backends.mps.is_available())

    model_dir: Path = args.model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    tokenizer_dir = resolve_tokenizer_dir(model_dir, args.tokenizer_dir)

    # Resolve device preference
    device = pick_device(None if args.device == "auto" else args.device)
    dtype_alias = pick_dtype(device, args.torch_dtype)
    # Map alias to torch dtype; keep string to pass to HF which accepts strings in new versions
    # Fall back to manual mapping for older versions
    alias_lower = dtype_alias.lower()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(alias_lower, torch.float32)

    logger.info(
        "bench_start | model_dir=%s | tokenizer=%s | device=%s | dtype=%s | greedy=%s | max_new=%d",
        str(model_dir), str(tokenizer_dir) if tokenizer_dir else "<auto>", device, torch_dtype, args.greedy, args.max_new_tokens,
    )

    # Seed for reproducibility if requested
    if args.seed is not None and torch is not None:
        try:
            import random
            random.seed(args.seed)
        except Exception:
            pass
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(
        str(tokenizer_dir or model_dir),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    t_tok = time.perf_counter() - t0
    logger.info("tokenizer_loaded | seconds=%.3f", t_tok)

    # Load model
    device_map = "auto" if device == "cuda" else None
    low_cpu_mem_usage = True
    t1 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    # Ensure evaluation mode for deterministic inference (disables dropout, etc.)
    model.eval()
    if device_map is None:
        # Move to single device if not using accelerate sharding
        target = torch.device(device)
        model.to(target)
    t_model = time.perf_counter() - t1
    logger.info("model_loaded | seconds=%.3f | device_map=%s", t_model, device_map or "<none>")

    # Prepare inputs
    prompt = args.prompt
    # Inject optional system reasoning/directive lines before the user prompt
    sys_prefix_parts = []
    if args.reasoning_level is not None:
        sys_prefix_parts.append(f"System: Reasoning: {args.reasoning_level}")
    if args.system_directive:
        sys_prefix_parts.append(f"System: {args.system_directive}")
    if sys_prefix_parts:
        prompt = "\n\n".join(sys_prefix_parts) + "\n\n" + prompt
    if args.think_prefix:
        prompt = f"{args.think_prefix}{prompt}"
    if args.think_suffix:
        prompt = f"{prompt}{args.think_suffix}"
    inputs = tok(prompt, return_tensors="pt")
    # Always move inputs to the model's primary device. Even with
    # device_map="auto" (single-GPU), HF does not automatically move
    # input tensors, which leads to device mismatch at embedding.
    try:
        model_device = model.device  # available on recent HF versions
    except Exception:
        # Fallback: infer from first parameter
        model_device = next(model.parameters()).device
    inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": not args.greedy,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "num_beams": int(args.num_beams),
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
    }
    if args.min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = int(args.min_new_tokens)
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(args.repetition_penalty)
    if args.no_repeat_ngram_size is not None:
        gen_kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)
    # Thread GPT-OSS specific kwargs through generate->forward
    if args.output_router_logits:
        gen_kwargs["output_router_logits"] = True
    if int(args.logits_to_keep) > 0:
        gen_kwargs["logits_to_keep"] = int(args.logits_to_keep)
    if args.no_use_cache:
        gen_kwargs["use_cache"] = False
    elif args.use_cache:
        gen_kwargs["use_cache"] = True

    # Generate
    torch.cuda.synchronize() if device == "cuda" else None
    t2 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize() if device == "cuda" else None
    t_gen = time.perf_counter() - t2

    # Decode and compute stats
    new_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    toks_per_s = float(new_tokens) / max(t_gen, 1e-9)
    decoded = tok.decode(out[0], skip_special_tokens=True)
    generated_text = decoded[len(prompt):]

    logger.info("generate_complete | seconds=%.3f | new_tokens=%d | toks_per_s=%.2f", t_gen, new_tokens, toks_per_s)
    if args.print_output:
        print("\n=== OUTPUT ===\n" + generated_text)

    # Optional manifest
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_dir": str(model_dir),
            "tokenizer_dir": str(tokenizer_dir) if tokenizer_dir else None,
            "device": device,
            "dtype": str(torch_dtype),
            "prompt": prompt,
            "max_new_tokens": int(args.max_new_tokens),
            "timings": {
                "tokenizer_load_s": t_tok,
                "model_load_s": t_model,
                "generate_s": t_gen,
            },
            "throughput": {
                "new_tokens": int(new_tokens),
                "tokens_per_s": toks_per_s,
            },
        }
        with args.manifest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("manifest_written | path=%s", str(args.manifest))


if __name__ == "__main__":  # pragma: no cover
    main()
