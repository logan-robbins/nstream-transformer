"""Shared Notes Cross-Attention layers used in the GPT-OSS integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(slots=True)
class SharedNotesCrossAttentionConfig:
    hidden_size: int
    notes_dim: int
    num_heads: int
    gating_init: float = -5.0


class SharedNotesCrossAttention(nn.Module):
    """Cross-attention that queries Dynamic Notes Bus snapshots."""

    def __init__(self, config: SharedNotesCrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_heads
        if head_dim * config.num_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.notes_dim, config.hidden_size)
        self.v_proj = nn.Linear(config.notes_dim, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.gate = nn.Parameter(torch.full((1,), config.gating_init))

    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        batch, sequence, _ = hidden_states.size()
        _, notes_len, _ = notes.size()
        if notes_len == 0:
            return hidden_states
        q = self.q_proj(hidden_states)
        k = self.k_proj(notes)
        v = self.v_proj(notes)
        q = q.view(batch, sequence, self.config.num_heads, -1).transpose(1, 2)
        k = k.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)
        v = v.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if notes_mask is not None:
            attn_scores = attn_scores.masked_fill(notes_mask[:, None, None, :] == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, sequence, -1)
        projected = self.o_proj(context)
        gating = torch.sigmoid(self.gate)
        return hidden_states + gating * projected


__all__ = ["SharedNotesCrossAttention", "SharedNotesCrossAttentionConfig"]
