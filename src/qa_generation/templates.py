"""Question templates specific to UK financial facts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QATemplate:
    """Describe a question/answer schema for UK fact benchmarking."""

    template_id: str
    question_text: str
    answer_expression: str  # expression evaluated against fact table


def load_default_templates() -> list[QATemplate]:
    """Yield baseline templates (e.g., revenue, net income) for UK filings."""
    raise NotImplementedError("Template authoring pending canonical fact definitions.")
