"""Load UK Companies House iXBRL documents and normalize context."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET
import pandas as pd


@dataclass(frozen=True)
class IXBRLDocument:
    """Encapsulates parsed facts, contexts, and metadata from one filing."""

    filing_id: str
    company_number: str
    facts: pd.DataFrame
    contexts: pd.DataFrame
    units: pd.DataFrame


def load_ixbrl(path: Path) -> IXBRLDocument:
    """Parse an iXBRL file from UK Companies House into a structured document."""
    if not path.exists():
        raise FileNotFoundError(f"iXBRL file does not exist: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    filing_id = path.stem
    company_number = path.parent.name

    facts_df = _extract_facts(root)
    contexts_df = _extract_contexts(root)
    units_df = _extract_units(root)

    return IXBRLDocument(
        filing_id=filing_id,
        company_number=company_number,
        facts=facts_df,
        contexts=contexts_df,
        units=units_df,
    )


def extract_contexts(document: IXBRLDocument) -> pd.DataFrame:
    """Return the unique contexts for further fact extraction."""
    return document.contexts.copy()

def extract_units(document: IXBRLDocument) -> pd.DataFrame:
    """Return the unit definitions mapping id -> measure text."""
    return document.units.copy()

def extract_facts(document: IXBRLDocument) -> pd.DataFrame:
    """Return the unique contexts for further fact extraction."""
    return document.facts.copy()

IX_NAMESPACES = (
    "http://www.xbrl.org/2013/inlineXBRL",
    "http://www.xbrl.org/2008/inlineXBRL",
)
XBRLI_NS = "http://www.xbrl.org/2003/instance"


def _text_content(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    text = "".join(element.itertext()).strip()
    return text or None

def _extract_facts(root: ET.Element) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for namespace in IX_NAMESPACES:
        for tag in ("nonFraction", "nonNumeric"):
            for fact in root.findall(f".//{{{namespace}}}{tag}"):
                text = _text_content(fact)
                if fact.tag.endswith("nonFraction") and fact.get("sign") == "-" and text:
                    if not text.startswith("-"):
                        text = f"-{text}"

                records.append(
                    {
                        "name": fact.get("name"),
                        "contextRef": fact.get("contextRef"),
                        "unitRef": fact.get("unitRef"),
                        "decimals": fact.get("decimals"),
                        "value": text,
                    }
                )

    if records:
        return pd.DataFrame.from_records(records)
    return pd.DataFrame(columns=["name", "contextRef", "unitRef", "decimals", "value"])

def _extract_contexts(root: ET.Element) -> pd.DataFrame:
    contexts: list[dict[str, Any]] = []

    for ctx in root.findall(f".//{{{XBRLI_NS}}}context"):
        context_id = ctx.get("id")
        entity_identifier = _text_content(
            ctx.find(f"./{{{XBRLI_NS}}}entity/{{{XBRLI_NS}}}identifier")
        )
        entity_scheme = None
        entity_node = ctx.find(f"./{{{XBRLI_NS}}}entity/{{{XBRLI_NS}}}identifier")
        if entity_node is not None:
            entity_scheme = entity_node.get("scheme")

        period_node = ctx.find(f"./{{{XBRLI_NS}}}period")
        start_date = _text_content(period_node.find(f"./{{{XBRLI_NS}}}startDate")) if period_node is not None else None
        end_date = _text_content(period_node.find(f"./{{{XBRLI_NS}}}endDate")) if period_node is not None else None
        instant = _text_content(period_node.find(f"./{{{XBRLI_NS}}}instant")) if period_node is not None else None

        scenario = ctx.find(f"./{{{XBRLI_NS}}}scenario")
        scenario_text = _text_content(scenario)

        contexts.append(
            {
                "context_id": context_id,
                "entity_identifier": entity_identifier,
                "entity_scheme": entity_scheme,
                "start_date": start_date,
                "end_date": end_date,
                "instant": instant,
                "scenario": scenario_text,
            }
        )

    if contexts:
        return pd.DataFrame.from_records(contexts)
    return pd.DataFrame(
        columns=[
            "context_id",
            "entity_identifier",
            "entity_scheme",
            "start_date",
            "end_date",
            "instant",
            "scenario",
        ]
    )

def _extract_units(root: ET.Element) -> pd.DataFrame:
    units: list[dict[str, Any]] = []
    for unit in root.findall(f".//{{{XBRLI_NS}}}unit"):
        unit_id = unit.get("id")
        measure = _text_content(unit.find(f"./{{{XBRLI_NS}}}measure"))
        units.append({"unit_id": unit_id, "measure": measure})
    if units:
        return pd.DataFrame.from_records(units)
    return pd.DataFrame(columns=["unit_id", "measure"])
