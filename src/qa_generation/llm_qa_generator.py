import os
import pandas as pd
from openai import OpenAI
import os
from pydantic import BaseModel, ValidationError
import json
from pathlib import Path

# Set OpenAI API key
api_key = Path(Path(__file__).resolve().parents[2]/"OPENAI_API_KEY").read_text().strip()
client = OpenAI(api_key=api_key)

def generate_qa_openai(input_csv_path, output_csv_path,
                         model="gpt-4o-mini"):
    
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
        entity = row.get('entity_name', 'Unknown Company')
        year = row.get('year', 'Unknown Year')
        metric = row.get('canonical_fact_name', 'Unknown Metric')
        value = row.get('ground_truth_value', 'N/A')

        # 4. The Optimized Prompt (9/10 Quality)
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
                temperature=0.7,       # Adds linguistic variety
                max_tokens=300,        # Limits cost (keeps answers concise)
                response_format={"type": "json_object"} # Guarantees strict JSON
            )

            # 6. Parse Response
            content = response.choices[0].message.content
            parsed_data = json.loads(content)

            # 7. Append to Results
            results.append({
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

