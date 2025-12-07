"""Validation checks for UK financial fact tables."""
from __future__ import annotations

import pandas as pd


def validate_fact_completeness(facts: pd.DataFrame) -> None:
    """Ensure required UK facts are available or raise informative errors."""
    raise NotImplementedError("Completeness checks will be tailored to UK reporting.")


def detect_consolidated_vs_individual(facts: pd.DataFrame) -> pd.Series:
    """Flag whether each fact row belongs to consolidated or individual statements."""
    raise NotImplementedError("Need to inspect context identifiers to infer consolidation level.")
