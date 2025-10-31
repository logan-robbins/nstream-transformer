"""Randomness helpers for deterministic experiments."""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_everything(seed: Optional[int], *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs when available.

    Parameters
    ----------
    seed:
        The seed to apply. When ``None`` the function is a no-op so callers
        can pass configuration values directly.
    deterministic:
        When ``True`` (default) additional PyTorch determinism flags are
        toggled to deliver reproducible kernels on CUDA-enabled hosts. The
        flag is ignored if PyTorch is not installed.
    """

    if seed is None:
        return

    value = int(seed)
    random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None:
        np.random.seed(value)  # type: ignore[attr-defined]

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None:
        torch.manual_seed(value)
        if torch.cuda.is_available():  # pragma: no cover - exercised on CUDA hosts
            torch.cuda.manual_seed_all(value)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:  # pragma: no cover - older PyTorch fallback
                pass
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - backend not available
                pass
