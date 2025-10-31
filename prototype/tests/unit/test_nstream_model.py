from __future__ import annotations

from types import SimpleNamespace

import torch

from nstream_transformer.models import NStreamModelConfig, NStreamTransformer


class FakeTrunk(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # Minimal parameter so the adapter's freeze policy has something to touch.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        use_cache: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        hidden = torch.zeros(input_ids.size(0), input_ids.size(1), self.hidden_size)
        return SimpleNamespace(hidden_states=[hidden, hidden])


def test_nstream_forward_outputs() -> None:
    config = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(config)
    model.trunk_adapter.attach_model(FakeTrunk(config.hidden_size))

    input_ids = torch.ones(2, 3, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    hidden_states = model.encode(input_ids, attention_mask=attention_mask)
    notes = torch.zeros(2, 2, config.notes_dim)
    notes_mask = torch.ones(2, 2, dtype=torch.long)
    role_ids = torch.tensor([0, 1])
    plan_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    plan_mask = torch.ones(2, 2, dtype=torch.bool)
    outputs = model(
        hidden_states,
        role=role_ids,
        notes=notes,
        notes_mask=notes_mask,
        plan_item_ids=plan_ids,
        plan_item_mask=plan_mask,
    )
    assert set(outputs.keys()) == {
        "planner_logits",
        "notes_logits",
        "speculative_notes",
        "agreement",
        "role_logits",
        "coverage_logits",
    }
    assert outputs["planner_logits"].shape[-1] == config.vocab_size
    assert outputs["coverage_logits"].shape[:2] == (2, 2)
