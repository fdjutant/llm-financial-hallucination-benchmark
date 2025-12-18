# Canonical financial facts extraction
import pandas as pd
from typing import Dict
import re

def create_gold_ground_truth(silver_df: pd.DataFrame, output_path):

    gold_df = silver_df[['entity_name', 'raw_name','segment','ground_truth_value',
                          'period_type', 'year']].copy()
    
    def _clean_raw_name(val):
        if pd.isna(val):
            return val
        s = str(val)
        # remove domain prefix if present
        if ":" in s:
            s = s.split(":", 1)[1]
        # normalize separators
        s = re.sub(r'[_\-\.\s]+', ' ', s)
        # split camelCase boundaries (aB -> a B) and XMLHttp style (XMLHttp -> XML Http)
        s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', s)
        s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s.title()

    gold_df.insert(3, 'canonical_fact_name',
                   gold_df['raw_name'].apply(_clean_raw_name))
    gold_df.drop(columns=['raw_name'], inplace=True)
    gold_df.to_csv(output_path, index=False)
    print(f"Gold data saved to: {output_path}")
    
    return gold_df

def create_flat_ground_truth(df: pd.DataFrame, output_path):
    # 1. Load Data
    # Convert 'dimensional_qualifier' to string immediately to avoid the "unhashable dict" error
    df['dimensional_qualifier'] = df['dimensional_qualifier'].astype(str)

    # 2. Create a Unified 'Answer' Column
    # If value_numeric exists, use it. If not, try to parse value_text as a float.
    # This handles cases where '29000000' is stored as text.
    def get_ground_truth_value(row):
        if pd.notna(row['value_numeric']):
            return row['value_numeric']
        
        # Try to convert text to number
        try:
            return float(row['value_text'])
        except (ValueError, TypeError):
            # If it's real text (like an accounting policy description), return that text
            return row['value_text']

    df['ground_truth_value'] = df.apply(get_ground_truth_value, axis=1)

    # 3. Create Clean Meta-Data Columns for Querying
    
    # Extract Year from period_end (useful for questions like "in 2023")
    df['period_end_dt'] = pd.to_datetime(df['period_end'], errors='coerce')
    df['period_start_dt'] = pd.to_datetime(df['period_start'], errors='coerce')
    df['year'] = df['period_end_dt'].dt.year

    # Clean up Entity Names (AstraZeneca_PLC_2023 -> AstraZeneca)
    # This helps when the LLM asks about "AstraZeneca" not "AstraZeneca_PLC_2023"
    def clean_entity(filing_id):
        if 'AstraZeneca' in str(filing_id): return 'AstraZeneca'
        if 'GSK' in str(filing_id): return 'GSK'
        return filing_id
    
    df['entity_name'] = df['filing_id'].apply(clean_entity)

    # 4. Add Financial Segment Classification
    def classify_financial_segment(raw_name: str) -> str:
        name = str(raw_name)

        # A. Narrative check
        if 'Explanatory' in name or 'Policy' in name or 'DisclosureOf' in name:
            return 'Narrative_Disclosure'

        # B. Company extension check
        if name.startswith(('azn:', 'gsk:')):
            return 'Company_Specific_Metric'

        # C. Financial statement logic
        lower = name.lower()
        if any(x in lower for x in ['cash', 'flow', 'activities']):
            return 'Cash_Flow'
        if any(x in lower for x in ['asset', 'liabilit', 'equity', 'inventory']):
            return 'Balance_Sheet'
        if any(x in lower for x in ['revenue', 'profit', 'loss', 'income', 'expense']):
            return 'Income_Statement'

        return 'Other_Financial_Metric'
    
    df['segment'] = df['raw_name'].apply(classify_financial_segment)

    # 5. Filter out rows that have NO value (empty text and empty numeric)
    df_clean = df[df['ground_truth_value'].notna()].copy()

    # 6. Save
    df_clean.to_csv(output_path, index=False)
    print(f"Created cleaned ground truth with {len(df_clean)} rows.")
    print(f"Silver data saved to: {output_path}")
    return df_clean

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

    df["domain"] = df["raw_name"].apply(get_domain)

    # --- gt_financial_facts ---
    gt_financial_facts = df[
        (df["domain"] == "frs-core") &
        (df["unit_ref"].fillna("").str.upper() == preferred_currency.upper()) &
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