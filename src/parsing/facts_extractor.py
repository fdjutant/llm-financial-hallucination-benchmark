# Canonical financial facts extraction
import pandas as pd
from typing import Dict

def extract_canonical_facts(facts_with_context: pd.DataFrame,
    preferred_currency: str = "GBP"
) -> Dict[str, pd.DataFrame]:
    """
    Given a raw iXBRL DataFrame, produce three ground truth tables:
      - financial_facts: numeric/monetary facts (frs-core, GBP)
      - narrative_policies: text/policy notes (frs-core, non-monetary)
      - entity_compliance: entity-level and compliance metadata (frs-bus, frs-direp)
    Returns a dict of DataFrames.
    """
    df = facts_with_context.copy()

    # Helper: extract domain from tag (e.g. "frs-core:PropertyPlantEquipment" -> "frs-core")
    def get_domain(tag):
        if pd.isnull(tag):
            return None
        return str(tag).split(":")[0] if ":" in str(tag) else None

    df["domain"] = df["name"].apply(get_domain)

    # --- gt_financial_facts ---
    gt_financial_facts = df[
        (df["domain"] == "frs-core") &
        (df["unitRef"].fillna("").str.upper() == preferred_currency.upper()) &
        (~df["value"].isin(["-", "", None]))
    ].copy()

    # --- gt_narrative_policies ---
    gt_narrative_policies = df[
        (df["domain"] == "frs-core") &
        (df["unitRef"].isnull()) &
        (df["value"].notnull())
    ].copy()

    # --- gt_entity_compliance ---
    gt_entity_compliance = df[
        df["domain"].isin(["frs-bus", "frs-direp"])
    ].copy()

    return {
        "financial_facts": gt_financial_facts,
        "narrative_policies": gt_narrative_policies,
        "entity_compliance": gt_entity_compliance,
    }

def extract_canonical_facts_dataframes(facts_with_context_dict):
    """Extract canonical facts for multiple filings."""
    canonical_facts_dict = {}
    for filing_id, df in facts_with_context_dict.items():
        canonical_facts_dict[filing_id] = extract_canonical_facts(df)
    return canonical_facts_dict

def save_canonical_facts_to_csv(canonical_facts_dict, output_dir):
    """
    Save canonical facts (financial_facts, narrative_policies, entity_compliance)
    for each filing_id to CSV files in the specified output directory.
    Each file will be named <filing_id>_<table_name>.csv.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    for filing_id, tables in canonical_facts_dict.items():
        for table_name, df in tables.items():
            csv_path = os.path.join(output_dir, f"{filing_id}_{table_name}.csv")
            df.to_csv(csv_path, index=False)