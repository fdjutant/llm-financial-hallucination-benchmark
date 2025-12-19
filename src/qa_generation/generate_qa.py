from pydantic import BaseModel
from pathlib import Path
import csv
import pandas as pd

class QAItem(BaseModel):
    question: str
    answer: str
    reasoning: str

def generate_question(row):
    entity = row['entity_name']
    year = row['year']
    metric = row['canonical_fact_name']
    value = row['ground_truth_value']
    question = (
        f"Given the financial statements and disclosures for {entity} in the fiscal year {year}, "
        f"calculate the reported value for '{metric}'. Consider all relevant adjustments, including any restatements, "
        f"non-recurring items, and footnote disclosures. What is the final value reported for this metric?"
    )
    answer = value
    reasoning = (
        f"To answer this question, the analyst must thoroughly review the financial statements, identify the '{metric}' metric, "
        f"and account for any adjustments or disclosures that may affect the reported value. The correct answer is '{value}', "
        f"as this is the value reported after considering all relevant factors."
    )
    return QAItem(question=question, answer=answer, reasoning=reasoning)

def generate_qa_pairs(input_csv_path, output_csv_path):

    input_csv_path = Path(input_csv_path)  # Ensure input_csv_path is a Path object
    output_csv_path = Path(output_csv_path)  # Ensure output_csv_path is a Path object
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output folder exists

    qa_items = []  # List to store QA items for DataFrame creation

    with input_csv_path.open(newline='', encoding='utf-8') as infile, output_csv_path.open('w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['question', 'answer', 'reasoning']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            qa = generate_question(row)
            writer.writerow(qa.model_dump())
            qa_items.append(qa.model_dump())  # Append the QA item as a dictionary

    # Create and return a DataFrame from the QA items
    return pd.DataFrame(qa_items)
