"""N-Stream Transformer package now wired onto GPT-OSS primitives."""

from importlib import import_module
from typing import Any

__all__ = ("config", "data", "inference", "integration", "models", "utils")


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
