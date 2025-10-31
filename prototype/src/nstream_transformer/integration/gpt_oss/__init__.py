"""GPT-OSS integration primitives for the N-Stream Transformer package."""

from .trunk_adapter import GptOssTrunkAdapter, TrunkAdapterConfig
from .embedder import GptOssEmbedder, GptOssEmbedderConfig

__all__ = [
    "GptOssTrunkAdapter",
    "TrunkAdapterConfig",
    "GptOssEmbedder",
    "GptOssEmbedderConfig",
]
