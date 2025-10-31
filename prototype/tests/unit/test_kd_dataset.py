from __future__ import annotations

import json

import torch

from nstream_transformer.training.dataset import KDJsonlDataset


def test_kd_dataset_decodes_snapshots(tmp_path) -> None:
    record = {
        "student_ids": [1, 2, 3],
        "student_labels": [1, 2, 3],
        "planner_ids": [4, 5, 6],
        "notes_student": [[0.1, 0.1], [0.2, 0.2]],
        "notes_teacher": [[0.9, 0.9], [1.0, 1.0]],
        "role": "core",
        "teacher_snapshots": [
            {
                "notes": [[0.9, 0.9], [1.0, 1.0]],
                "stride": 0,
                "version": 1,
                "role": "core",
                "coverage_flags": [1, 0],
            }
        ],
        "student_snapshots": [
            {
                "notes": [[0.3, 0.3], [0.4, 0.4]],
                "stride": 0,
                "version": 1,
                "role": "core",
            }
        ],
        "agreement_labels": [1],
        "stride_ids": [0],
        "commit_points": [2],
        "plan_tokens": [
            "intro::set the context",
            "core::Discuss tactics and criticisms",
            "wrap::Summarise key takeaways",
        ],
        "notes_tokens": ["discuss", "core", "tactics"],
        "example_id": "abc-123",
        "metadata": {
            "document_text": "Intro paragraph.\n\nCore paragraph describing tactics.",
            "document_paragraphs": ["Intro paragraph.", "Core paragraph describing tactics."],
            "teacher_plan": {
                "plan": [
                    {"role": "intro", "summary": "Intro summary", "notes": ["set context"]},
                    {"role": "core", "summary": "Core summary", "notes": ["Discuss tactics and criticisms"]},
                    {"role": "wrap", "summary": "Wrap summary", "notes": ["Summarise key takeaways"]},
                ],
                "segments": [
                    {"role": "intro", "paragraph_start": 0, "paragraph_end": 1},
                    {"role": "core", "paragraph_start": 1, "paragraph_end": 2},
                    {"role": "wrap", "paragraph_start": 2, "paragraph_end": 3},
                ],
            },
            "teacher_notes": {
                "intro": ["set context"],
                "core": ["Discuss tactics and criticisms"],
                "wrap": ["Summarise key takeaways"],
            },
        },
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = KDJsonlDataset(path)
    example = dataset[0]

    assert example["student_ids"].dtype == torch.long
    assert example["notes_teacher"].shape == torch.Size([2, 2])
    assert len(example["teacher_snapshots"]) == 1
    snapshot = example["teacher_snapshots"][0]
    assert snapshot.notes.shape == torch.Size([2, 2])
    assert example["agreement_labels"].shape == torch.Size([1])
    assert example["example_id"] == "abc-123"
    assert example["plan_items"] == ["Discuss tactics and criticisms", "Core summary"]
    assert example["plan_catalog"] == [
        "set context",
        "Intro summary",
        "Discuss tactics and criticisms",
        "Core summary",
        "Summarise key takeaways",
        "Wrap summary",
    ]
    assert example["plan_catalog_roles"] == ["intro", "intro", "core", "core", "wrap", "wrap"]
    assert example["coverage_targets"] == [0, 0, 1, 1, 0, 0]
    assert "tactics" in example["notes_text"]
    assert example["raw_teacher_notes"]["core"] == ["Discuss tactics and criticisms"]
