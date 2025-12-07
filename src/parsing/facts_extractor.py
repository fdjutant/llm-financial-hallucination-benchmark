# Canonical financial facts extraction
import pandas as pd
from typing import Dict, List, Any

def tag_mapping() -> Dict[str, List[str]]:
    """
    Returns a mapping of canonical financial metrics to lists of iXBRL tags (including synonyms).
    """
    return {
        "property_plant_equipment": [
            "core:PropertyPlantEquipment", "frs-core:PropertyPlantEquipment"
        ],
        "fixed_assets": [
            "core:FixedAssets", "frs-core:FixedAssets"
        ],
        "debtors": [
            "core:Debtors", "frs-core:Debtors"
        ],
        "cash_bank_on_hand": [
            "core:CashBankOnHand", "frs-core:CashBankOnHand"
        ],
        "current_assets": [
            "core:CurrentAssets", "frs-core:CurrentAssets"
        ],
        "creditors": [
            "core:Creditors", "frs-core:Creditors"
        ],
        "net_current_assets_liabilities": [
            "core:NetCurrentAssetsLiabilities", "frs-core:NetCurrentAssetsLiabilities"
        ],
        "total_assets_less_current_liabilities": [
            "core:TotalAssetsLessCurrentLiabilities", "frs-core:TotalAssetsLessCurrentLiabilities"
        ],
        "net_assets_liabilities": [
            "core:NetAssetsLiabilities", "frs-core:NetAssetsLiabilities"
        ],
        "equity": [
            "core:Equity", "frs-core:Equity"
        ],
        # Add more as needed for revenue, profit before tax, net income, total liabilities, etc.
    }

def extract_canonical_facts(
    facts_with_context: pd.DataFrame,
    tag_map: Dict[str, List[str]],
    preferred_currency: str = "GBP",
    prefer_consolidated: bool = True,
    fiscal_year: Any = None
) -> pd.DataFrame:
    """
    Extracts canonical financial facts (e.g. revenue, profit before tax, net income, total assets, total liabilities)
    from a merged iXBRL facts DataFrame, using tag mapping, context, and unit filtering.

    Args:
        facts_with_context: DataFrame with iXBRL facts merged with context and units.
        tag_map: Dict mapping canonical metric names to lists of iXBRL tags (including synonyms).
        preferred_currency: Only facts in this currency are returned (default: GBP).
        prefer_consolidated: If True, prefer consolidated over individual statements when both exist.
        fiscal_year: Optionally filter for a specific fiscal year (context or period).

    Returns:
        DataFrame with one row per canonical fact, including value, context, and metadata.
    """

    print("TESTING extract_canonical_facts function")

    results = []

    # Use actual column names from your CSVs
    tag_col = 'name'
    currency_col = 'unitRef' #if 'unitRef' in facts_with_context.columns else 'measure'
    context_col = 'contextRef' #if 'contextRef' in facts_with_context.columns else 'context_id'
    date_col = 'instant' #if 'end_date' in facts_with_context.columns else 'instant'

    print("DEBUG: Unique tag values in DataFrame:", facts_with_context[tag_col].unique())
    for metric, tags in tag_map.items():

        print(f"\nDEBUG: Filtering for metric '{metric}' with tags {tags}")

        # Filter by tag (case-insensitive, strip whitespace)
        tag_mask = facts_with_context[tag_col].str.strip().str.lower().isin([t.lower() for t in tags])
        filtered = facts_with_context[tag_mask]
        print(f"DEBUG: Rows after tag filter: {len(filtered)}")

        # Filter by currency
        if currency_col in filtered.columns:
            filtered = filtered[filtered[currency_col].str.upper() == preferred_currency.upper()]
            print(f"DEBUG: Rows after currency filter: {len(filtered)}")

        # Filter by fiscal year (end_date)
        if fiscal_year and date_col in filtered.columns:
            filtered = filtered[filtered[date_col] == fiscal_year]
            print(f"DEBUG: Rows after fiscal year filter: {len(filtered)}")

        # Select the most recent or relevant fact
        if not filtered.empty:

            # Sort by date_col if available
            if date_col in filtered.columns:
                best_fact = filtered.sort_values(date_col, ascending=False).iloc[0]
            else:
                best_fact = filtered.iloc[0]
            print(f"DEBUG: Selected fact for '{metric}':", best_fact.to_dict())

            results.append({
                'metric': metric,
                'value': best_fact['value'],
                'currency': best_fact.get(currency_col, None),
                'period_end': best_fact.get(date_col, None),
                'context_id': best_fact.get(context_col, None),
                'source_tag': best_fact[tag_col],
                'raw_row': best_fact.to_dict(),
            })
        else:
            print(f"DEBUG: No fact found for '{metric}' after filtering.")
            results.append({
                'metric': metric,
                'value': None,
                'currency': None,
                'period_end': None,
                'context_id': None,
                'source_tag': None,
                'raw_row': None,
            })

    return pd.DataFrame(results)