"""Utilities for working with UK Companies House filing downloads."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import pdb

import datetime as dt
import base64
import requests
import os
import json
from IPython.display import JSON, display

BASE_URL = "https://api.company-information.service.gov.uk"
API_KEY = Path(Path(__file__).resolve().parents[2]/"COMPANIES_HOUSE_API_KEY").read_text().strip()
auth = base64.b64encode(f"{API_KEY}:".encode()).decode()
headers = {"Authorization": f"Basic {auth}"}
RAW_COMPANIES_HOUSE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "companies_house"
MANIFEST_PATH = RAW_COMPANIES_HOUSE_DIR / "manifest.json"
PREFERRED_IXBRL_MIMETYPE = "application/xhtml+xml"

@dataclass(frozen=True)
class FilingRequest:
    """Describe a single filing download request for a UK company."""

    company_number: str
    company_name: Optional[str]
    filing_id: str
    filing_date: dt.date
    destination: Path
    document_url: str

def search_company(name):
    param = {"q": name}
    r = requests.get(f"{BASE_URL}/search/companies", 
                     headers=headers, params=param)
    r.raise_for_status()
    return r.json()

def fetch_active_company_numbers(
    query: str, *, items_per_page: int = 100, max_pages: int = 5
) -> list[str]:
    """Return company numbers for search hits whose ``company_status`` is ``active``."""

    # TO DO: Too many irrelevant companies - may need stringent query search

    if not query or not query.strip():
        raise ValueError("Companies House search API requires a non-empty `query` string.")

    active_numbers: list[str] = []
    start_index = 0

    for _ in range(max_pages):
        params = {
            "q": query,
            "items_per_page": items_per_page,
            "start_index": start_index,
        }

        response = requests.get(
            f"{BASE_URL}/search/companies", headers=headers, params=params
        )
        response.raise_for_status()
        data = response.json()

        debug_path = RAW_COMPANIES_HOUSE_DIR / "debug" / "last_search.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        for item in data.get("items", []):
            if item.get("company_status") == "active" and item.get("company_number"):
                active_numbers.append(item["company_number"])

        total_results = data.get("total_results")
        start_index += items_per_page

        if not data.get("items"):
            break

        if isinstance(total_results, int) and start_index >= total_results:
            break

    return active_numbers

def build_download_manifest(
    company_numbers: Iterable[str], *,
    manifest_path: Optional[Path] = RAW_COMPANIES_HOUSE_DIR/"manifest.json"
) -> list[FilingRequest]:
    """Return manifest entries (iXBRL only) and update on-disk manifest for traceability."""

    manifest: list[FilingRequest] = []
    manifest_path = manifest_path or MANIFEST_PATH
    manifest_index = _load_manifest_index(manifest_path)

    for company_number in company_numbers:
        profile_response = requests.get(
            f"{BASE_URL}/company/{company_number}",
            headers=headers,
        )
        profile_response.raise_for_status()
        company_name = profile_response.json().get("company_name")

        response = requests.get(
            f"{BASE_URL}/company/{company_number}/filing-history",
            headers=headers,
        )
        response.raise_for_status()
        filings = response.json()
        
        # debug output
        debug_path = RAW_COMPANIES_HOUSE_DIR / "debug" / "all_filings_from_chosen_companies.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(filings, f, indent=2)

        for item in filings.get("items", []):
            if item.get("category") != "accounts":
                continue

            description = (item.get("description") or "").lower()
            if "full" not in description and "group" not in description:
                continue

            metadata_link = item.get("links", {}).get("document_metadata")
            if not metadata_link:
                continue

            try:
                metadata = _fetch_document_metadata(metadata_link)
            except requests.HTTPError as exc:
                print(f"Skipping filing {company_number}:{metadata_link} ({exc})")
                continue

            resources = metadata.get("resources") or {}
            if PREFERRED_IXBRL_MIMETYPE not in resources:
                # Skip filings where only PDF renditions are present.
                continue

            date_str = item.get("date")
            try:
                filing_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue

            filing_id = item.get("transaction_id") or item.get("barcode")
            if not filing_id:
                filing_id = f"{company_number}-{date_str}"

            destination = RAW_COMPANIES_HOUSE_DIR / company_number / f"{filing_id}.ixbrl"
            
            manifest.append(
                FilingRequest(
                    company_number=company_number,
                    company_name=company_name,
                    filing_id=filing_id,
                    filing_date=filing_date,
                    destination=destination,
                    document_url=metadata_link,
                )
            )

            manifest.sort(key=lambda req: req.filing_date, reverse=True)

            manifest_entry = manifest_index.get(filing_id, {}).copy()
            manifest_entry.update(
                {
                    "company_name": company_name,
                    "company_number": company_number,
                    "filing_id": filing_id,
                    "made_up_date": item.get("description_values", {}).get("made_up_date"),
                    "source_url": metadata_link,
                }
            )
            download_ts = _existing_download_timestamp(destination) or manifest_entry.get(
                "download_timestamp"
            )
            manifest_entry["download_timestamp"] = download_ts
            manifest_index[filing_id] = manifest_entry

    _write_manifest(manifest_index, manifest_path)
    return manifest 

def download_ixbrl_filing(request: FilingRequest) -> Path:
    """ Download an IXBRL filing for UK Companies House """
   
    metadata_url = request.document_url
    metadata_response = requests.get(metadata_url, headers=headers)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    year = metadata["created_at"][:10]

    document_url = metadata.get("links", {}).get("document")

    # debug output
    debug_path = RAW_COMPANIES_HOUSE_DIR / "debug" / "individual_document_metadata.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not document_url:
        raise ValueError("Document URL is not available")

    preferred_type = "application/xhtml+xml"
    resources = metadata.get("resources") or {}

    destination = request.destination   
    destination.parent.mkdir(parents=True, exist_ok = True)

    download_headers = dict(headers)
    download_headers["Accept"] = preferred_type

    with requests.get(document_url, headers=download_headers, stream=True) as resp:
        resp.raise_for_status()

        if resp.headers["Content-Type"] != "application/xhtml+xml":
            raise ValueError(f"Content-Type is not expected: {resp.headers['Content-Type']}")

        with destination.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"{year}, {resp.headers['Content-Type']}, {destination}")

    return destination

def _load_manifest_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Return manifest entries keyed by filing id (preserving download timestamps)."""

    if not manifest_path.exists():
        return {}

    try:
        manifest_doc = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}

    entries = manifest_doc.get("entries", [])
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        filing_id = entry.get("filing_id")
        if filing_id:
            index[filing_id] = entry
    return index

def _manifest_entry_sort_key(entry: dict[str, Any]) -> tuple[str, int, str]:
    """Sort companies alphabetically and filings by most recent made_up_date first."""

    company_number = entry.get("company_number") or ""
    made_up_date = entry.get("made_up_date")
    try:
        ordinal = dt.datetime.strptime(made_up_date, "%Y-%m-%d").date().toordinal()
    except (TypeError, ValueError):
        ordinal = dt.date.min.toordinal()
    filing_id = entry.get("filing_id") or ""
    return (company_number, -ordinal, filing_id)

def _format_timestamp_from_datetime(value: dt.datetime) -> str:
    """Return ISO timestamp (UTC, second precision) for ``value``."""

    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _format_timestamp(timestamp: float) -> str:
    """Return ISO timestamp string for a POSIX timestamp (UTC)."""

    dt_value = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    return _format_timestamp_from_datetime(dt_value)

def _existing_download_timestamp(destination: Path) -> Optional[str]:
    """If an iXBRL file already exists locally, return its last modification timestamp."""

    if not destination.exists():
        return None
    return _format_timestamp(destination.stat().st_mtime)

def _write_manifest(index: dict[str, dict[str, Any]], manifest_path: Path) -> None:
    """Persist manifest entries to disk in deterministic order."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_entries = sorted(index.values(), key=_manifest_entry_sort_key)
    manifest_doc = {
        "generated_at": _format_timestamp_from_datetime(dt.datetime.now(tz=dt.timezone.utc)),
        "entries": ordered_entries,
    }
    manifest_path.write_text(json.dumps(manifest_doc, indent=2))

def _fetch_document_metadata(metadata_url: str) -> dict[str, Any]:
    """Retrieve Companies House document metadata JSON."""

    response = requests.get(metadata_url, headers=headers)
    response.raise_for_status()
    return response.json()
