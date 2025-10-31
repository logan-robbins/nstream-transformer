from __future__ import annotations

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from nstream_transformer.data.teacher_provider import TeacherNotes, TeacherNotesProviderBase
from nstream_transformer.data.teacher_runner import TeacherRunnerConfig
from nstream_transformer.models import NStreamModelConfig, NStreamTransformer
from nstream_transformer.training import Trainer, TrainingConfig
from nstream_transformer.training import trainer as trainer_module
from nstream_transformer.training.trainer import StagePolicyConfig


class _StaticTeacherNotesProvider(TeacherNotesProviderBase):
    """Deterministic teacher provider for unit tests."""

    def __init__(self, *_, **__):
        pass

    def fetch(self, example: dict[str, object]) -> TeacherNotes:
        notes = example.get("notes_teacher")
        if not isinstance(notes, torch.Tensor):
            notes = torch.tensor(notes, dtype=torch.float32)
        return TeacherNotes(notes=notes.clone(), snapshots=[], raw_notes={})


trainer_module.TeacherNotesProvider = _StaticTeacherNotesProvider


def _training_config(**kwargs: object) -> TrainingConfig:
    cfg = TrainingConfig(**kwargs)
    cfg.teacher_runner = TeacherRunnerConfig(provider="mock", model="mock-model")
    cfg.device = "cpu"
    return cfg


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, _: int) -> dict[str, object]:
        return {
            "student_ids": torch.tensor([1, 2, 3]),
            "student_labels": torch.tensor([1, 2, 3]),
            "planner_ids": torch.tensor([4, 5, 6]),
            "notes_student": torch.zeros(3, 4),
            "notes_teacher": torch.zeros(3, 4),
            "plan_items": ["Cover the topic"],
            "coverage_targets": [1],
            "notes_text": "cover topic",
            "plan_tokens": ["core::Cover the topic"],
            "notes_tokens": ["cover", "topic"],
            "role": "core",
        }


class FakeTrunk(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        use_cache: bool = False,
        **_: object,
    ) -> object:
        hidden = torch.zeros(input_ids.size(0), input_ids.size(1), self.hidden_size)
        return type("Outputs", (), {"hidden_states": [hidden, hidden]})()


def test_trainer_runs_single_step() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=1, log_interval=1, eval_interval=10)
    dataset = FakeDataset()
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=dataset,
        eval_dataset=None,
    )
    trainer.fit()
    assert trainer.state.global_step == 1


def test_gradnorm_adjusts_scale() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.gradnorm.enabled = True
    trainer_cfg.gradnorm.target_ratio = 1.0
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    initial_scale = trainer.kd_scale
    trainer._maybe_adjust_gradnorm(torch.tensor(2.0), torch.tensor(1.0), stage=2)
    assert trainer.kd_scale < initial_scale
    decreased_scale = trainer.kd_scale
    trainer._maybe_adjust_gradnorm(torch.tensor(0.2), torch.tensor(1.0), stage=2)
    assert trainer.kd_scale > decreased_scale


def test_interhead_spec_kl_matches_expected() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    speculative = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32, device=trainer.device)
    mask = torch.tensor([[1, 1]], dtype=torch.long, device=trainer.device)

    loss = trainer._interhead_spec_kl(speculative, mask, temperature=1.0)

    log_probs = F.log_softmax(speculative[0], dim=-1)
    probs = log_probs.exp()
    kl_01 = torch.sum(probs[0] * (log_probs[0] - log_probs[1]))
    kl_10 = torch.sum(probs[1] * (log_probs[1] - log_probs[0]))
    expected = (kl_01 + kl_10) / 1.0

    assert torch.isclose(loss, expected, atol=1e-6)


def test_interhead_spec_kl_respects_overlap_threshold() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    speculative = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32, device=trainer.device)
    mask = torch.tensor([[1, 1]], dtype=torch.long, device=trainer.device)
    coverage = torch.tensor([[1.0, 1e-6]], dtype=torch.float32, device=trainer.device)

    loss = trainer._interhead_spec_kl(
        speculative,
        mask,
        temperature=1.0,
        coverage=coverage,
        min_overlap=1e-4,
    )

    assert torch.isclose(loss, torch.tensor(0.0, device=trainer.device), atol=1e-8)


def test_stability_and_rollback_metrics_emitted() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.curriculum.L = 1
    trainer_cfg.metrics.stability_every = 1
    dataset = FakeDataset()
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    batch = trainer.collator([dataset[0], dataset[1]])
    _, metrics = trainer._training_step(batch)

    assert "rollback_ratio" in metrics
    assert "stability_ratio" in metrics
    assert metrics.get("rollback_tokens", 0.0) > 0.0
    assert metrics.get("stability_tokens", 0.0) > 0.0
    assert "repair_error_rate" in metrics
    assert "stability_error_rate" in metrics


def test_stage_schedule_advances_as_configured() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.curriculum.steps_per_stage = 0
    trainer_cfg.curriculum.stage_schedule = (0, 2, 5, 7)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer.state.global_step = 0
    assert trainer._determine_stage() == 0
    trainer.state.global_step = 2
    assert trainer._determine_stage() == 1
    trainer.state.global_step = 5
    assert trainer._determine_stage() == 2
    trainer.state.global_step = 9
    assert trainer._determine_stage() == 3


def test_stage_policy_freeze_unfreeze_applied() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.curriculum.stage_schedule = (0, 1)
    trainer_cfg.stage_policies = {
        0: StagePolicyConfig(freeze=("trunk",)),
        1: StagePolicyConfig(unfreeze=("trunk",)),
    }
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer.state.global_step = 0
    trainer._determine_stage()
    assert all(not param.requires_grad for param in model.trunk_adapter.model.parameters())

    trainer.state.global_step = 1
    trainer._determine_stage()
    assert any(param.requires_grad for param in model.trunk_adapter.model.parameters())


def test_negative_sampling_augments_plan_items_and_notes() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.negative_sampling.enabled = True
    trainer_cfg.negative_sampling.start_stage = 0
    trainer_cfg.negative_sampling.contradiction_ratio = 1.0
    trainer_cfg.negative_sampling.max_contradictions = 2
    trainer_cfg.negative_sampling.noise_ratio = 1.0
    trainer_cfg.negative_sampling.noise_std = 0.5
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    dataset = FakeDataset()
    batch = trainer.collator([dataset[0], dataset[1]])
    original_width = batch["plan_item_ids"].shape[1]
    original_notes = batch["notes_student"].clone()

    trainer._maybe_apply_negative_sampling(batch, stage=1)

    assert batch["plan_item_ids"].shape[1] > original_width
    neg_targets = batch["coverage_targets"][:, original_width:]
    assert neg_targets.numel() > 0
    assert torch.all(neg_targets == 0.0)
    assert torch.any(batch["notes_student"] != original_notes)
    if "plan_text" in batch:
        assert len(batch["plan_text"][0]) > len(dataset[0]["plan_items"])


def test_micro_rollout_updates_bus_and_commit_mask() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.parallel_micro_steps = 2
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )
    trainer.state.stage_index = 2

    dataset = FakeDataset()
    batch_cpu = trainer.collator([dataset[0], dataset[1]])
    batch = {
        key: value.to(trainer.device) if torch.is_tensor(value) else value for key, value in batch_cpu.items()
    }

    hidden_states = trainer.model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])
    teacher_branch = trainer._prepare_branch_inputs(batch, branch="teacher", stage=2)
    student_branch = trainer._prepare_branch_inputs(batch, branch="student", stage=2, teacher_branch=teacher_branch)

    initial_bus_mask = batch["student_bus_mask"].clone()
    initial_commit_mask = batch["commit_mask"].clone()

    plan_item_ids = batch.get("plan_item_ids")
    plan_item_mask = batch.get("plan_item_mask")

    _ = trainer._run_student_pass(
        hidden_states,
        batch,
        student_branch,
        stage=2,
        plan_item_ids=plan_item_ids,
        plan_item_mask=plan_item_mask,
    )

    updated_bus_mask = batch["student_bus_mask"]
    assert updated_bus_mask.sum().item() >= initial_bus_mask.sum().item()
    assert not torch.equal(initial_commit_mask, batch["commit_mask"].cpu())


def test_agreement_labels_autogenerated_when_missing() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.loss_weights.agree = 1.0
    trainer_cfg.curriculum.L = 2
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer.state.stage_index = 4
    dataset = FakeDataset()
    batch = trainer.collator([dataset[0], dataset[1]])

    _, metrics = trainer._training_step(batch)

    assert "agreement_loss" in metrics
    assert "agreement_precision" in metrics
    assert metrics.get("agreement_auto") == 1.0


def test_generate_training_report_summarises_history() -> None:
    model_cfg = NStreamModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = NStreamTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer._log_metrics("train", {"loss": 1.0, "mask_ablation": 0.5, "kd_ce_ratio": 0.2})
    trainer.state.global_step = 5
    trainer._log_metrics(
        "train",
        {"loss": 0.8, "mask_ablation": 0.4, "kd_ce_ratio": 0.3, "agreement_precision": 0.9},
    )
    trainer._log_metrics("eval", {"eval_loss": 0.75})

    report = trainer.generate_training_report()

    assert report["global_step"] == trainer.state.global_step
    assert report["train_history_length"] == 2
    mask_summary = report["train_metrics"]["mask_ablation"]
    assert mask_summary["last"] == 0.4
    assert mask_summary["min"] == 0.4
    assert mask_summary["max"] == 0.5
    assert "eval_loss" in report["eval_metrics"]
