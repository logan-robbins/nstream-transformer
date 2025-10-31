"""Integration tests for the multi-role inference orchestrator."""

from __future__ import annotations

import types
from typing import Dict, Iterable, Tuple

import torch
from torch import nn

from nstream_transformer.inference import (
    DecodeConfig,
    GateAnnealingConfig,
    InferenceConfig,
    MultiStreamOrchestrator,
)


class FakeTokenizer:
    def __call__(self, text: str, *, return_tensors: str = "pt", add_special_tokens: bool = True):
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(65 + (int(token_id) % 26)) for token_id in token_ids)


class FakeTrunkModel(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            values = torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size)
            self.embed.weight.copy_(values / hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        for i in range(min(hidden_size, vocab_size)):
            self.lm_head.weight.data[i, i] = 1.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        hidden = self.proj(self.embed(input_ids))
        seq_len = hidden.size(1)
        if past_key_values is None:
            zero = torch.zeros(hidden.size(0), seq_len, hidden.size(-1), device=hidden.device)
            past_key_values = ((zero.clone(), zero.clone()),)
        return types.SimpleNamespace(hidden_states=[hidden], past_key_values=past_key_values)


class FakeTrunkAdapter:
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        self.model = FakeTrunkModel(hidden_size, vocab_size)

    def load_model(self) -> None:  # pragma: no cover - compatibility hook
        return None


class FakeRoleAdapters(nn.Module):
    def __init__(self, roles: Tuple[str, ...], hidden_size: int) -> None:
        super().__init__()
        self.role_to_index = {role: idx for idx, role in enumerate(roles)}
        self.bias = nn.Parameter(torch.zeros(len(roles), hidden_size))

    def forward(self, role: str, hidden_states: torch.Tensor) -> torch.Tensor:
        idx = self.role_to_index[role.lower()]
        bias = self.bias[idx].view(1, 1, -1)
        return hidden_states + bias


class FakeCrossAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if notes.size(1) == 0:
            return hidden_states
        summary = notes.mean(dim=1, keepdim=True)
        return hidden_states + summary


class FakeSpeculationHead(nn.Module):
    def forward(self, adapted: torch.Tensor) -> torch.Tensor:
        return adapted.clone()


class FakeNotesHead(nn.Module):
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history.clone()


class FakeAgreementHead(nn.Module):
    def __init__(self, scores: Iterable[float]) -> None:
        super().__init__()
        self.register_buffer("_scores", torch.tensor(list(scores), dtype=torch.float32))
        self._index = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._scores.numel() == 0:
            value = 1.0
        else:
            value = float(self._scores[self._index % self._scores.numel()].item())
        self._index += 1
        return hidden_states.new_full((hidden_states.size(0), 1, 1), value)


class FakePlannerHead(nn.Module):
    def __init__(self, hidden_size: int, plan_vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, plan_vocab_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class FakePlanEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, plan_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = plan_ids.shape
        return torch.zeros((batch, seq_len, self.hidden_size), dtype=torch.float32, device=plan_ids.device)


class FakeCoverageHead(nn.Module):
    def forward(
        self,
        attended: torch.Tensor,
        plan_embeddings: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = plan_embeddings.size(0), plan_embeddings.size(1)
        return attended.new_zeros(batch, seq_len)


class FakeModel(nn.Module):
    def __init__(self, roles: Tuple[str, ...]) -> None:
        super().__init__()
        hidden_size = 4
        vocab_size = 32
        self.roles = roles
        self.config = types.SimpleNamespace(
            notes_dim=hidden_size,
            notes_head=types.SimpleNamespace(notes_dim=hidden_size),
            notes_bus=types.SimpleNamespace(dtype="float32"),
        )
        self.trunk_adapter = FakeTrunkAdapter(hidden_size, vocab_size)
        self.role_adapters = FakeRoleAdapters(roles, hidden_size)
        self.cross_attention = FakeCrossAttention()
        self.speculation_head = FakeSpeculationHead()
        self.notes_head = FakeNotesHead()
        self.agreement_head = FakeAgreementHead([0.0, 0.8, 0.8, 0.8, 0.8])
        self.planner_head = FakePlannerHead(hidden_size, vocab_size)
        self.plan_embedding = FakePlanEmbedding(hidden_size=hidden_size)
        self.coverage_head = FakeCoverageHead()


def _assert_stride_pattern(events, role: str, stride: int) -> None:
    role_events = [item for item in events if item.role == role]
    stride_indices = [event.stride_index for event in role_events]
    assert stride_indices == sorted(stride_indices)
    counts: Dict[int, int] = {}
    for event in role_events:
        counts[event.stride_index] = counts.get(event.stride_index, 0) + 1
        assert counts[event.stride_index] <= stride


def test_orchestrator_emits_notes_and_rolls_back_on_low_agreement() -> None:
    torch.manual_seed(0)
    roles = ("intro", "core", "wrap")
    config = InferenceConfig(
        roles=roles,
        stride_B=2,
        commit_L=4,
        read_lag_delta=0,
        max_snapshots_K=4,
        gate_g=1.0,
        agreement_threshold_tau=0.5,
        emission_cadence_M_by_role={role: 1 for role in roles},
        decode=DecodeConfig(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=3,
            do_sample=False,
            seed=123,
        ),
        gate_annealing=GateAnnealingConfig(enabled=False),
        rng_seed=123,
    )

    tokenizer = FakeTokenizer()
    model = FakeModel(roles)
    orchestrator = MultiStreamOrchestrator(model, tokenizer, config)
    orchestrator.start("test prompt")

    events = []
    while True:
        outcome = orchestrator.step()
        if outcome is None:
            break
        events.append(outcome)

    manifest = orchestrator.finalize()

    assert len(events) == manifest["steps"]
    assert any(event.rollback_performed for event in events)
    assert manifest["rollbacks"], "Expected rollback events recorded in manifest."

    for role in roles:
        _assert_stride_pattern(events, role, config.stride_B)

    bus_versions: Dict[str, int] = {
        role: orchestrator.bus_by_role[role].latest_version() for role in roles
    }
    assert bus_versions["intro"] >= 3

    for role, state in orchestrator.states.items():
        for producer, version in state.last_seen_version.items():
            assert version <= bus_versions[producer]

    rollback_roles = {record["role"] for record in manifest["rollbacks"]}
    assert "intro" in rollback_roles
