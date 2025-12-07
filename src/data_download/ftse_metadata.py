"""Maintain metadata for UK-focused company universe (e.g., FTSE constituents)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CompanyProfile:
    """Minimal metadata for a UK company we plan to benchmark."""

    name: str
    ticker: str
    company_number: str
    fiscal_year_end: str  # e.g., "31-12"


def load_ftse_universe() -> list[CompanyProfile]:
    """Return the curated starter universe for FTSE benchmarking."""

    seed_data = [
        {
            "name": "Finance Advice Centre Ltd",
            "ticker": "FAC-PRIVATE",
            "company_number": "08948140",
            "fiscal_year_end": "31-03",
        },
    ]

    return [CompanyProfile(**record) for record in seed_data]

