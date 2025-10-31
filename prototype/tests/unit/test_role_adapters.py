from __future__ import annotations

import torch

from nstream_transformer.models import RoleAdapterConfig, RoleAdapters


def test_role_adapter_single_role() -> None:
    config = RoleAdapterConfig(hidden_size=8, bottleneck_size=2, roles=("intro",))
    adapters = RoleAdapters(config)
    hidden = torch.randn(1, 3, 8)
    output = adapters("intro", hidden)
    assert output.shape == hidden.shape


def test_role_adapter_batch_roles() -> None:
    config = RoleAdapterConfig(hidden_size=8, bottleneck_size=2)
    adapters = RoleAdapters(config)
    hidden = torch.randn(2, 3, 8)
    role_ids = torch.tensor([0, 1])
    output = adapters(role_ids, hidden)
    assert output.shape == hidden.shape
