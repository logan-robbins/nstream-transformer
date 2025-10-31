#!/usr/bin/env python
"""Generate intro/core/wrap plan and sections using the teacher LLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure src/ is on path for local execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from nstream_transformer.data.extraction import TeacherPlanner


def load_article(source_path: Path, title: str | None = None, article_id: str | None = None) -> tuple[str, str]:
    """Load article title and text from a JSONL or raw text file."""
    if source_path.suffix == ".txt":
        text = source_path.read_text(encoding="utf-8")
        resolved_title = title or source_path.stem
        return resolved_title, text

    # Assume JSONL with fields {"title": ..., "text": ..., "id": ...}
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload: dict[str, Any] = json.loads(line)
            if title and payload.get("title") != title:
                continue
            if article_id and str(payload.get("id")) != article_id:
                continue
            if not title and not article_id:
                # default to first record if no filter provided
                pass
            return payload.get("title", "Untitled"), payload.get("text", "")

    raise FileNotFoundError(
        f"Article not found in {source_path} (title={title}, id={article_id})"
    )


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, collapsing consecutive blank lines."""
    raw_parts = text.replace("\r", "").split("\n\n")
    paragraphs = [part.strip() for part in raw_parts if part.strip()]
    if not paragraphs:
        return [text.strip()]
    return paragraphs


def format_section(name: str, paragraphs: Sequence[str], start: int, end: int) -> str:
    header = f"==== {name.upper()} (paragraphs {start}–{end}) ===="
    body = "\n\n".join(paragraphs[start:end])
    return f"{header}\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to raw text (.txt) or JSONL article file")
    parser.add_argument("--title", type=str, default=None, help="Article title (for JSONL sources)")
    parser.add_argument("--id", type=str, default=None, help="Article id (for JSONL sources)")
    parser.add_argument("--model", type=str, default="llama3.1:70b-instruct-q5_K_M", help="Teacher model name")
    parser.add_argument("--provider", type=str, default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file to write teacher response and segments",
    )
    args = parser.parse_args()

    article_title, article_text = load_article(args.source, title=args.title, article_id=args.id)
    if not article_text.strip():
        raise SystemExit("Article text is empty")

    paragraphs = split_paragraphs(article_text)
    planner = TeacherPlanner(provider=args.provider, model=args.model)
    teacher_plan = planner.build_plan(article_title, paragraphs)

    print(f"Article: {article_title}\nTotal paragraphs: {len(paragraphs)}\n")
    print("Plan:")
    for segment in teacher_plan.segments:
        summary = teacher_plan.get_summary(segment.role) or ""
        print(
            f" - {segment.role.lower():<5} paragraphs {segment.paragraph_start}–{segment.paragraph_end}: {summary}"
        )

    segments_output: list[str] = []
    for segment in teacher_plan.segments:
        section_text = format_section(
            segment.role,
            paragraphs,
            segment.paragraph_start,
            segment.paragraph_end,
        )
        segments_output.append(section_text)

    print("\nSegments written to output file." if segments_output else "No segments found.")

    # Determine output path
    if args.output is not None:
        output_path = args.output
    else:
        title_slug = re.sub(r"[^a-z0-9]+", "-", article_title.lower()).strip("-") or "article"
        output_dir = _REPO_ROOT / "experiments" / "teacher_plans"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{title_slug}.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan_lines = ["# Plan", ""]
    for entry in teacher_plan.summaries:
        plan_lines.append(f"- {entry.role.lower()}: {entry.summary}")
    plan_lines.append("")

    report_lines = [
        "# Teacher Response",
        json.dumps(teacher_plan.raw, indent=2, ensure_ascii=False),
        "",
    ]
    report_lines.extend(plan_lines)
    report_lines.extend(["# Segments", ""])
    report_lines.extend(segments_output)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
