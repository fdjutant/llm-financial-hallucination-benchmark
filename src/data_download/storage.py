"""Storage helpers for UK filing downloads and manifests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class StorageLayout:
    """Describes where UK filings and derived assets live locally."""

    raw_dir: Path
    manifest_path: Path


def ensure_layout(layout: StorageLayout) -> None:
    """Create directories/manifests for UK filings if they do not exist."""
    layout.raw_dir.mkdir(parents=True, exist_ok=True)
    if not layout.manifest_path.exists():
        layout.manifest_path.write_text(json.dumps({"files": []}, indent=2))


def register_file(layout: StorageLayout, filing_id: str, relative_path: str) -> None:
    """Register a filing inside the manifest; keeps UK data reproducible."""
    manifest = json.loads(layout.manifest_path.read_text())
    manifest.setdefault("files", []).append({"filing_id": filing_id, "path": relative_path})
    layout.manifest_path.write_text(json.dumps(manifest, indent=2))
