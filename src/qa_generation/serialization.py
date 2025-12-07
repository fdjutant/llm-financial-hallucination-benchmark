"""Serialize QA datasets with reproducibility metadata."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Sequence

from .builder import QAPair


def write_qa_dataset(pairs: Sequence[QAPair], path: Path) -> None:
    """Write QA pairs plus metadata to disk for UK benchmark runs."""
    payload = {
        "pairs": [pair.__dict__ for pair in pairs],
        "metadata": {"jurisdiction": "UK", "version": 0},
    }
    path.write_text(json.dumps(payload, indent=2))
