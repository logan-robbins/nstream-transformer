"""Role adherence head supervising role-specific representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class RoleClassifierConfig:
    hidden_size: int
    num_roles: int
    dropout: float = 0.0


class RoleClassifierHead(nn.Module):
    def __init__(self, config: RoleClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_roles)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        features = torch.tanh(self.proj(self.dropout(pooled)))
        return self.classifier(features)


__all__ = ["RoleClassifierConfig", "RoleClassifierHead"]
