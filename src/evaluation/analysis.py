"""Downstream analysis helpers for UK QA benchmark results."""
from __future__ import annotations

import pandas as pd


def summarize_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize accuracy metrics grouped by question template or company."""
    raise NotImplementedError("Pending result schema definition.")
