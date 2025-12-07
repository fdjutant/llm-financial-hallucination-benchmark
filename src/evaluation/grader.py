"""Grading utilities comparing LLM answers to UK fact ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GradingResult:
    """Captures how an answer compared to the UK ground truth."""

    status: str  # correct / approximate / wrong / hallucinated
    notes: str | None = None


class AnswerParser(Protocol):
    """Protocol for parsing numeric answers from free-form text."""

    def parse_numeric(self, answer: str) -> float | None:
        ...


def grade_answer(
    model_answer: str,
    ground_truth: float,
    parser: AnswerParser,
    relative_tolerance: float = 0.02,
) -> GradingResult:
    """Grade a numeric answer according to tolerance bands."""
    raise NotImplementedError("Need parsing heuristics aligned with UK facts before grading.")
