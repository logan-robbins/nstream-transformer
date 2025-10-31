"""N-Stream Transformer wired directly onto GPT-OSS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from ..data.collator_kd import TwoBranchKDCollatorConfig
from ..inference.dnb_bus import DynamicNotesBus, DynamicNotesBusConfig
from ..inference.snc_cross_attn import SharedNotesCrossAttention, SharedNotesCrossAttentionConfig
from ..integration.gpt_oss import GptOssTrunkAdapter, TrunkAdapterConfig
from .heads import (
    AgreementHead,
    AgreementHeadConfig,
    CoverageHead,
    CoverageHeadConfig,
    NotesHead,
    NotesHeadConfig,
    PlannerHead,
    PlannerHeadConfig,
    RoleClassifierConfig,
    RoleClassifierHead,
    SpeculationHead,
    SpeculationHeadConfig,
)
from .role_adapters import RoleAdapterConfig, RoleAdapters


@dataclass(slots=True)
class NStreamModelConfig:
    hidden_size: int = 4096
    vocab_size: int = 32000
    notes_dim: int = 2048
    num_heads: int = 32
    plan_vocab_size: int = 65536
    trunk: TrunkAdapterConfig = field(default_factory=TrunkAdapterConfig)
    role_adapters: Optional[RoleAdapterConfig] = None
    notes_bus: Optional[DynamicNotesBusConfig] = None
    cross_attention: Optional[SharedNotesCrossAttentionConfig] = None
    planner_head: Optional[PlannerHeadConfig] = None
    notes_head: Optional[NotesHeadConfig] = None
    speculation_head: Optional[SpeculationHeadConfig] = None
    agreement_head: Optional[AgreementHeadConfig] = None
    coverage_head: Optional[CoverageHeadConfig] = None
    role_classifier_head: Optional[RoleClassifierConfig] = None
    collator: TwoBranchKDCollatorConfig = field(
        default_factory=lambda: TwoBranchKDCollatorConfig(pad_token_id=0)
    )

    def __post_init__(self) -> None:
        if self.role_adapters is None:
            self.role_adapters = RoleAdapterConfig(
                hidden_size=self.hidden_size,
                bottleneck_size=self.hidden_size // 8,
            )
        if self.notes_bus is None:
            self.notes_bus = DynamicNotesBusConfig(snapshot_dim=self.notes_dim)
        if self.cross_attention is None:
            self.cross_attention = SharedNotesCrossAttentionConfig(
                hidden_size=self.hidden_size,
                notes_dim=self.notes_dim,
                num_heads=self.num_heads,
            )
        if self.planner_head is None:
            self.planner_head = PlannerHeadConfig(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
            )
        if self.notes_head is None:
            self.notes_head = NotesHeadConfig(
                hidden_size=self.hidden_size,
                notes_dim=self.notes_dim,
            )
        if self.speculation_head is None:
            self.speculation_head = SpeculationHeadConfig(
                hidden_size=self.hidden_size,
                notes_dim=self.notes_dim,
            )
        if self.agreement_head is None:
            self.agreement_head = AgreementHeadConfig(hidden_size=self.hidden_size)
        if self.coverage_head is None:
            self.coverage_head = CoverageHeadConfig(hidden_size=self.hidden_size)
        if self.role_classifier_head is None:
            self.role_classifier_head = RoleClassifierConfig(
                hidden_size=self.hidden_size,
                num_roles=len(self.collator.role_to_id),
            )
        if self.collator.notes_dim != self.notes_dim:
            self.collator = TwoBranchKDCollatorConfig(
                pad_token_id=self.collator.pad_token_id,
                label_pad_id=self.collator.label_pad_id,
                notes_dim=self.notes_dim,
                max_length=self.collator.max_length,
                role_to_id=self.collator.role_to_id,
                plan_hash_buckets=self.collator.plan_hash_buckets,
                dtype=self.collator.dtype,
            )


class NStreamTransformer(nn.Module):
    """Destructive rewrite of the model stack around GPT-OSS."""

    def __init__(self, config: NStreamModelConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk_adapter = GptOssTrunkAdapter(config.trunk)
        self.role_adapters = RoleAdapters(config.role_adapters)  # type: ignore[arg-type]
        self.notes_bus = DynamicNotesBus(config.notes_bus)  # type: ignore[arg-type]
        self.cross_attention = SharedNotesCrossAttention(config.cross_attention)  # type: ignore[arg-type]
        self.planner_head = PlannerHead(config.planner_head)  # type: ignore[arg-type]
        self.notes_head = NotesHead(config.notes_head)  # type: ignore[arg-type]
        self.speculation_head = SpeculationHead(config.speculation_head)  # type: ignore[arg-type]
        self.agreement_head = AgreementHead(config.agreement_head)  # type: ignore[arg-type]
        self.coverage_head = CoverageHead(config.coverage_head)  # type: ignore[arg-type]
        self.role_classifier = RoleClassifierHead(config.role_classifier_head)  # type: ignore[arg-type]
        self.plan_embedding = nn.Embedding(config.plan_vocab_size, config.hidden_size)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        trunk = self.trunk_adapter.model
        outputs = trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **generation_kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        role: torch.Tensor | str,
        notes: torch.Tensor,
        notes_mask: Optional[torch.Tensor] = None,
        plan_item_ids: Optional[torch.Tensor] = None,
        plan_item_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        adapted = self.role_adapters(role, hidden_states)
        attended = self.cross_attention(adapted, notes, notes_mask=notes_mask)
        # Align training with inference by computing planner logits from the
        # notes-conditioned (attended) states. This makes mask-ablation and
        # stability diagnostics meaningful and encourages notes sensitivity.
        planner_logits = self.planner_head(attended)
        notes_logits = self.notes_head(attended)
        speculative_notes = self.speculation_head(adapted)
        agreement_score = self.agreement_head(attended).squeeze(-1)
        role_logits = self.role_classifier(adapted)
        coverage_logits: Optional[torch.Tensor] = None
        if plan_item_ids is not None and plan_item_mask is not None:
            embedded_plan = self.plan_embedding(plan_item_ids)
            plan_mask_bool = plan_item_mask.to(dtype=torch.bool, device=embedded_plan.device)
            coverage_logits = self.coverage_head(attended, embedded_plan, plan_mask_bool)
        return {
            "planner_logits": planner_logits,
            "notes_logits": notes_logits,
            "speculative_notes": speculative_notes,
            "agreement": agreement_score,
            "role_logits": role_logits,
            "coverage_logits": coverage_logits,
        }

    def iter_trainable_parameters(self):
        yield from self.trunk_adapter.iter_trainable_parameters()
        yield from self.role_adapters.parameters()
        yield from self.cross_attention.parameters()
        yield from self.planner_head.parameters()
        yield from self.notes_head.parameters()
        yield from self.speculation_head.parameters()
        yield from self.agreement_head.parameters()
        yield from self.coverage_head.parameters()
        yield from self.role_classifier.parameters()
        yield from self.plan_embedding.parameters()

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Collect only adapter/head parameters for lightweight checkpointing.

        Excludes the GPT-OSS trunk weights; intended for PEFT-style fine-tunes.
        """
        payload: Dict[str, torch.Tensor] = {}
        modules = {
            "role_adapters": self.role_adapters,
            "cross_attention": self.cross_attention,
            "planner_head": self.planner_head,
            "notes_head": self.notes_head,
            "speculation_head": self.speculation_head,
            "agreement_head": self.agreement_head,
            "coverage_head": self.coverage_head,
            "role_classifier": self.role_classifier,
            "plan_embedding": self.plan_embedding,
        }
        for prefix, module in modules.items():
            for name, tensor in module.state_dict().items():
                payload[f"{prefix}.{name}"] = tensor.detach().cpu()
        return payload

    def load_adapters(self, state_dict: Dict[str, torch.Tensor], *, strict: bool = False) -> None:
        """Load adapter/head parameters from a checkpoint mapping.

        Args:
            state_dict: mapping of parameter names to tensors
            strict: if True, require exact key matches within adapter modules
        """
        modules = {
            "role_adapters": self.role_adapters,
            "cross_attention": self.cross_attention,
            "planner_head": self.planner_head,
            "notes_head": self.notes_head,
            "speculation_head": self.speculation_head,
            "agreement_head": self.agreement_head,
            "coverage_head": self.coverage_head,
            "role_classifier": self.role_classifier,
            "plan_embedding": self.plan_embedding,
        }
        missing_total: list[str] = []
        unexpected_total: list[str] = []
        for prefix, module in modules.items():
            scoped = {k[len(prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(prefix + ".")}
            if not scoped:
                continue
            missing, unexpected = module.load_state_dict(scoped, strict=strict)
            missing_total.extend([f"{prefix}.{k}" for k in missing])
            unexpected_total.extend([f"{prefix}.{k}" for k in unexpected])
        if strict and (missing_total or unexpected_total):
            raise RuntimeError(
                f"Adapter load failed. Missing={missing_total} Unexpected={unexpected_total}"
            )


__all__ = ["NStreamTransformer", "NStreamModelConfig"]
