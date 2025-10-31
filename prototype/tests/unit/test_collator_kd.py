from __future__ import annotations

import torch

from nstream_transformer.data.collator_kd import (
    TwoBranchKDCollatorConfig,
    TwoBranchKnowledgeDistillationCollator,
)


def test_two_branch_collator_shapes() -> None:
    config = TwoBranchKDCollatorConfig(pad_token_id=0, notes_dim=2)
    collator = TwoBranchKnowledgeDistillationCollator(config)
    batch = [
        {
            "student_ids": torch.tensor([1, 2, 3]),
            "student_labels": torch.tensor([1, 2, 3]),
            "planner_ids": torch.tensor([4, 5]),
            "notes_student": torch.ones(2, 2),
            "notes_teacher": torch.ones(2, 2),
            "teacher_snapshots": [
                {"notes": [[1.0, 1.0], [1.0, 1.0]], "stride": 0, "version": 0, "role": "core"}
            ],
            "student_snapshots": [
                {"notes": [[0.5, 0.5], [0.5, 0.5]], "stride": 0, "version": 0, "role": "core"}
            ],
            "agreement_labels": [1],
            "plan_items": ["Discuss tactics"],
            "coverage_targets": [1],
            "notes_text": "Discuss tactics",
            "role": "core",
        },
        {
            "student_ids": torch.tensor([1, 2]),
            "student_labels": torch.tensor([1, 2]),
            "planner_ids": torch.tensor([6]),
            "notes_student": torch.ones(2, 2),
            "notes_teacher": torch.ones(2, 2),
            "teacher_snapshots": [
                {"notes": [[1.0, 1.0], [1.0, 1.0]], "stride": 0, "version": 0, "role": "intro"}
            ],
            "student_snapshots": [
                {"notes": [[0.5, 0.5], [0.5, 0.5]], "stride": 0, "version": 0, "role": "intro"}
            ],
            "agreement_labels": [0],
            "plan_items": ["Provide overview"],
            "coverage_targets": [0],
            "notes_text": "intro overview",
            "role": "intro",
        },
    ]
    output = collator(batch)
    assert output["input_ids"].shape[0] == 2
    assert output["notes_student"].shape[-1] == 2
    assert torch.all(output["role_ids"] >= 0)
    assert output["teacher_notes_bus"].shape[0] == 2
    assert output["teacher_notes_bus"].shape[1] == config.max_snapshots
    assert output["teacher_notes_bus"].shape[2] == 2
    assert output["commit_mask"].dtype == torch.bool
    assert output["agreement_labels"].shape == torch.Size([2, config.max_snapshots])
    assert output["coverage_targets"].shape == torch.Size([2, 1])
    assert output["plan_item_ids"].shape[0] == 2
    assert output["plan_item_roles"].shape == output["plan_item_ids"].shape
    assert output["plan_text"][0] == ["Discuss tactics"]
