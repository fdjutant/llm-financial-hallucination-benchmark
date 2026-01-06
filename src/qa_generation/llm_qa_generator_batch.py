import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI


# ---------- OpenAI client setup ----------
# Reads API key from API_KEY/OPENAI_API_KEY (same pattern as llm_qa_generator.py)
API_KEY_PATH = Path(__file__).resolve().parents[2] / "API_KEY" / "OPENAI_API_KEY"
API_KEY = API_KEY_PATH.read_text().strip() if API_KEY_PATH.exists() else os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)


EXCLUDED_SEGMENTS = ["Narrative_Disclosure", "Other_Financial_Metric"]


def _build_prompt(entity: str, year: Any, metric: str, value: Any) -> str:
    return (
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


def prepare_jsonl(
    input_csv_path: str,
    output_jsonl_path: str,
    mapping_csv_path: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 300,
) -> int:
    """Reads the input CSV, filters rows, builds JSONL batch requests and a mapping CSV.

    Returns the number of requests written.
    """
    df = pd.read_csv(input_csv_path)

    df_processing = df[~df["segment"].isin(EXCLUDED_SEGMENTS)].copy()

    # Ensure output directories exist
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Build lines and mapping
    lines: List[str] = []
    mapping_rows: List[Dict[str, Any]] = []

    for idx, row in df_processing.iterrows():
        row_id = row.get("id", idx)
        entity = row.get("entity_name", "Unknown Company")
        year = row.get("year", "Unknown Year")
        metric = row.get("canonical_fact_name", "Unknown Metric")
        value = row.get("ground_truth_value", "N/A")

        custom_id = f"qa_{row_id}"

        prompt = _build_prompt(entity, year, metric, value)

        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful financial analyst assistant. Output valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        one = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        lines.append(json.dumps(one))
        mapping_rows.append(
            {
                "custom_id": custom_id,
                "id": row_id,
                "entity_name": entity,
                "year": year,
                "segment": row["segment"],
                "original_metric": metric,
                "ground_truth_value": value,
            }
        )

    # Write JSONL
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    # Write mapping CSV
    pd.DataFrame(mapping_rows).to_csv(mapping_csv_path, index=False)

    print(f"Prepared {len(lines)} batch requests to {output_jsonl_path}")
    print(f"Wrote mapping to {mapping_csv_path}")
    return len(lines)


def submit_batch(input_jsonl_path: str, completion_window: str = "24h") -> str:
    """Uploads the JSONL and creates a Batch job. Returns the batch id."""
    # Upload file for batch purpose
    with open(input_jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )

    print(f"Submitted batch: id={batch.id}, status={getattr(batch, 'status', 'unknown')}")
    return batch.id


def check_status(batch_id: str) -> Dict[str, Any]:
    """Retrieves and prints batch status. Returns the batch dict."""
    batch = client.batches.retrieve(batch_id)

    status = getattr(batch, "status", None)
    request_counts = getattr(batch, "request_counts", None)
    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)

    print(f"Batch {batch_id} status: {status}")
    if request_counts:
        print(f"Counts: {request_counts}")
    if output_file_id:
        print(f"Output file id: {output_file_id}")
    if error_file_id:
        print(f"Error file id: {error_file_id}")

    # Convert to plain dict for ease of consumption
    try:
        batch_dict = batch.model_dump()  # pydantic style in SDK
    except Exception:
        # Fallback: best-effort
        batch_dict = {
            "id": getattr(batch, "id", None),
            "status": status,
            "request_counts": request_counts,
            "output_file_id": output_file_id,
            "error_file_id": error_file_id,
        }
    return batch_dict


def collect_results(
    batch_id: str,
    output_jsonl_path: str,
    mapping_csv_path: str,
    results_csv_path: str,
) -> int:
    """Downloads batch output JSONL, parses it, and writes the final results CSV.

    Returns number of parsed rows.
    """
    batch = client.batches.retrieve(batch_id)
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("Batch output not available yet. Status: " + str(getattr(batch, "status", None)))

    # Download output content
    content_stream = client.files.content(output_file_id)
    # Some SDKs return a stream-like object; get bytes
    try:
        data_bytes = content_stream.read()
    except AttributeError:
        # New SDK returns a Response with .text
        data_text = getattr(content_stream, "text", None)
        if data_text is None:
            # Try bytes
            data_bytes = getattr(content_stream, "content", b"")
        else:
            data_bytes = data_text.encode("utf-8")

    data_text = data_bytes.decode("utf-8")

    # Save raw output JSONL for reference
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        f.write(data_text)
    print(f"Saved batch output JSONL to {output_jsonl_path}")

    # Load mapping
    mapping_df = pd.read_csv(mapping_csv_path)
    mapping = {row["custom_id"]: row for _, row in mapping_df.iterrows()}

    # Parse each line
    results: List[Dict[str, Any]] = []
    for line in data_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)

        custom_id = obj.get("custom_id")
        error = obj.get("error")
        response = obj.get("response")

        if error:
            # Record failure row with error message for traceability
            base = mapping.get(custom_id, {})
            results.append(
                {
                    "id": base.get("id"),
                    "entity_name": base.get("entity_name"),
                    "year": base.get("year"),
                    "segment": base.get("segment"),
                    "original_metric": base.get("original_metric"),
                    "ground_truth_value": base.get("ground_truth_value"),
                    "generated_question": None,
                    "generated_answer": None,
                    "generated_reasoning": f"ERROR: {error}",
                }
            )
            continue

        if not response:
            continue

        body = response.get("body", {})
        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except Exception:
            parsed = {"question": None, "answer": None, "reasoning": None}

        base = mapping.get(custom_id, {})
        results.append(
            {
                "id": base.get("id"),
                "entity_name": base.get("entity_name"),
                "year": base.get("year"),
                "segment": base.get("segment"),
                "original_metric": base.get("original_metric"),
                "ground_truth_value": base.get("ground_truth_value"),
                "generated_question": parsed.get("question"),
                "generated_answer": parsed.get("answer"),
                "generated_reasoning": parsed.get("reasoning"),
            }
        )

    # Write final CSV
    Path(results_csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_csv_path, index=False)
    print(f"Wrote parsed results to {results_csv_path} ({len(results)} rows)")
    return len(results)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=300)


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch runner for QA generation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prepare
    p_prepare = sub.add_parser("prepare", help="Build JSONL requests and mapping CSV")
    p_prepare.add_argument("input_csv")
    p_prepare.add_argument("output_jsonl")
    p_prepare.add_argument("mapping_csv")
    _add_common_args(p_prepare)

    # submit
    p_submit = sub.add_parser("submit", help="Submit a batch job from JSONL")
    p_submit.add_argument("input_jsonl")
    p_submit.add_argument("--window", default="24h", choices=["24h", "48h"])

    # status
    p_status = sub.add_parser("status", help="Check a batch job status")
    p_status.add_argument("batch_id")

    # collect
    p_collect = sub.add_parser("collect", help="Download output and write results CSV")
    p_collect.add_argument("batch_id")
    p_collect.add_argument("output_jsonl")
    p_collect.add_argument("mapping_csv")
    p_collect.add_argument("results_csv")

    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare_jsonl(
            input_csv_path=args.input_csv,
            output_jsonl_path=args.output_jsonl,
            mapping_csv_path=args.mapping_csv,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    elif args.cmd == "submit":
        submit_batch(args.input_jsonl, completion_window=args.window)
    elif args.cmd == "status":
        check_status(args.batch_id)
    elif args.cmd == "collect":
        collect_results(
            batch_id=args.batch_id,
            output_jsonl_path=args.output_jsonl,
            mapping_csv_path=args.mapping_csv,
            results_csv_path=args.results_csv,
        )


if __name__ == "__main__":
    main()
