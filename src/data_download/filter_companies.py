from pathlib import Path
import json
import pandas as pd

def filter_ftse100(result):
    """
    Filters FTSE100 companies based on specific criteria and saves the filtered data to a JSON file.

    Args:
        result (dict): Dictionary containing company data.
        project_root (Path): Path to the project root directory.

    Returns:
        pd.DataFrame: DataFrame containing filtered company data.
    """
    filtered_result = {
        key: min(
            (item for item in value.get("items", [])
             if item.get("company_type") == "plc"
             and item.get("company_status") == "active"),
            key=lambda x: x.get("date_of_creation", ""),
            default=None,
        )
        for key, value in result.items()
    }

    return filtered_result

def create_filtered_dataframe(filtered_result):
    """
    Creates a DataFrame from the filtered company data.

    Args:
        filtered_result (dict): Dictionary containing filtered company data.

    Returns:
        pd.DataFrame: DataFrame containing the filtered company data.
    """
    return pd.DataFrame([
        {
            'title': value.get('title'),
            'company_number': value.get('company_number'),
            'date_of_creation': value.get('date_of_creation')
        }
        for value in filtered_result.values() if value is not None
    ])