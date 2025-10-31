"""Role adapters applied to the upper transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass(slots=True)
class RoleAdapterConfig:
    hidden_size: int
    bottleneck_size: int = 1024
    roles: tuple[str, ...] = ("intro", "core", "wrap")
    activation: str = "gelu"
    dropout: float = 0.0


class _AdapterBlock(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        if activation == "relu":
            act: nn.Module = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.GELU()
        self.activation = act
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = hidden_states
        hidden = self.down(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.up(hidden)
        hidden = self.dropout(hidden)
        hidden = residual + hidden
        return self.layer_norm(hidden)


class RoleAdapters(nn.Module):
    """Container mapping role identifiers to independent adapter blocks."""

    def __init__(self, config: RoleAdapterConfig) -> None:
        super().__init__()
        self.config = config
        modules = {
            role: _AdapterBlock(
                hidden_size=config.hidden_size,
                bottleneck_size=config.bottleneck_size,
                activation=config.activation,
                dropout=config.dropout,
            )
            for role in config.roles
        }
        self.adapters = nn.ModuleDict(modules)

    def forward(self, role: torch.Tensor | str, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if isinstance(role, torch.Tensor):
            roles = role.tolist()
            outputs = []
            for index, role_id in enumerate(roles):
                role_name = self._role_name(role_id)
                adapter = self.adapters[role_name]
                outputs.append(adapter(hidden_states[index : index + 1]))
            return torch.cat(outputs, dim=0)
        try:
            adapter = self.adapters[role]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown role adapter requested: {role!r}") from exc
        return adapter(hidden_states)

    @property
    def available_roles(self) -> tuple[str, ...]:
        return tuple(self.adapters.keys())

    def state_dict_shallow(self) -> Dict[str, torch.Tensor]:
        """Collect a flattened state dict for PEFT snapshots."""

        payload: Dict[str, torch.Tensor] = {}
        for name, module in self.adapters.items():
            for param_name, tensor in module.state_dict().items():
                key = f"{name}.{param_name}"
                payload[key] = tensor.detach().cpu()
        return payload

    def _role_name(self, index: int) -> str:
        try:
            return self.config.roles[index]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Role index {index} is outside configured roles") from exc


__all__ = ["RoleAdapterConfig", "RoleAdapters"]
