"""Build programmatically verifiable QA pairs from UK filings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .templates import QATemplate


@dataclass(frozen=True)
class QAPair:
    """Stores a single generated QA example with provenance."""

    question: str
    answer: str
    source_filing_id: str
    template_id: str


def generate_qa_pairs(facts: pd.DataFrame, templates: Iterable[QATemplate]) -> list[QAPair]:
    """Generate QA pairs for UK filings using canonical facts and templates."""
    raise NotImplementedError("Need fact schema before implementing generation logic.")
