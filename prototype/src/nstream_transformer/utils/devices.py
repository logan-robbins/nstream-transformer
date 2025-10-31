"""Device utilities with macOS MPS/CUDA support and CPU fallback."""

from __future__ import annotations

import os
from typing import Literal


def resolve_device() -> Literal["cpu", "cuda", "mps"]:
    """Resolve the runtime device.

    Prefers CUDA, then Apple Metal (MPS), otherwise CPU. Respects `NSTREAM_DEVICE`.
    """

    forced = os.getenv("NSTREAM_DEVICE")
    if forced in {"cpu", "cuda", "mps"}:
        return forced  # honour explicit override

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Metal Performance Shaders (MPS) backend
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
