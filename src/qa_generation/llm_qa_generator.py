import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI


# ---------- OpenAI client setup ----------
API_KEY_PATH = Path(__file__).resolve().parents[2] / "API_KEY" / "OPENAI_API_KEY"
API_KEY = API_KEY_PATH.read_text().strip() if API_KEY_PATH.exists() else os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)


def load_config(config_path: str) -> dict:
    """Load and return a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_qa_openai(
    input_csv_path: str,
    output_csv_path: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 300,
):
    
    # 1. Read the Data
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(df)} rows from {input_csv_path}")
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # 2. Filter Data (As discussed: Numeric segments only)
    # We exclude 'Narrative_Disclosure' (text) and 'Other_Financial_Metric' (mixed/noisy)
    excluded_segments = ['Narrative_Disclosure', 'Other_Financial_Metric']
    df_processing = df[~df['segment'].isin(excluded_segments)].copy()
    
    results = []

    # 3. Iterate and Generate
    for idx, row in df_processing.iterrows():

        # Extract context variables
        id = row.get('id', 'N/A')
        entity = row.get('entity_name', 'Unknown Company')
        year = row.get('year', 'Unknown Year')
        metric = row.get('canonical_fact_name', 'Unknown Metric')
        value = row.get('ground_truth_value', 'N/A')

        # 4. The Prompt 
        prompt = (
            f"You are creating a financial benchmark dataset.\n"
            f"Context: In {year}, {entity} reported a '{metric}' of {value}.\n\n"
            "TASK:\n"
            f"1. Write a specific, natural-language question asking for the '{metric}' for {entity} in {year}.\n"
            "2. Write a 'Reasoning' step. Since you do not have the full report text, the reasoning should simply state: "
            f"'The specific metric {metric} for {entity} in {year} is explicitly reported as {value}.' "
            "but vary the phrasing slightly so it does not look like a robot template.\n"
            f"3. Provide the exact Answer (must match '{value}').\n\n"
            "OUTPUT FORMAT: JSON with keys 'question', 'answer', 'reasoning'."
        )

        try:
            # 5. The API Call (Optimized for Cost & Diversity)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst assistant. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            # 6. Parse Response
            content = response.choices[0].message.content
            parsed_data = json.loads(content)

            # 7. Append to Results
            results.append({
                "id": id,
                "entity_name": entity,
                "year": year,
                "segment": row['segment'],
                "original_metric": metric,
                "ground_truth_value": value,
                "generated_question": parsed_data.get("question"),
                "generated_answer": parsed_data.get("answer"),
                "generated_reasoning": parsed_data.get("reasoning")
            })

            # Optional: Progress indicator every 10 rows
            if (len(results) % 10) == 0:
                print(f"Processed {len(results)} rows...")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Continue to next row even if one fails
            continue

    # 8. Save Results
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! Generated {len(output_df)} QA pairs.")
    print(f"Saved to: {output_csv_path}")
    
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Serial QA generator (row-by-row, no Batch API)")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("input_csv", nargs="?", help="Input gold CSV path (manual mode)")
    parser.add_argument("output_csv", nargs="?", help="Output QA pairs CSV path (manual mode)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=300)
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        generate_qa_openai(
            input_csv_path=config["input"]["gold_csv"],
            output_csv_path=config["output"]["qa_pairs_csv"],
            model=config["model"]["model_name"],
            temperature=config["model"].get("temperature", 0.7),
            max_tokens=config["model"].get("max_tokens", 300),
        )
    else:
        if not args.input_csv or not args.output_csv:
            parser.error("Provide --config, or supply both input_csv and output_csv positional arguments")
        generate_qa_openai(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
