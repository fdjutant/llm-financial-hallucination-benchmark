import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from openai import OpenAI

from llm_interface import generate_document, modify_value


# ---------- API key paths ----------
API_KEY_DIR = Path(__file__).resolve().parents[2] / "API_KEY"


def get_client_and_provider(provider: str) -> Tuple[OpenAI, str]:
    """Get the appropriate OpenAI client and provider name based on the provider string.
    
    Args:
        provider: One of 'openai', 'groq', 'nebius'
        
    Returns:
        Tuple of (client, provider_name)
    """
    if provider == "openai":
        api_key_path = API_KEY_DIR / "OPENAI_API_KEY"
        api_key = api_key_path.read_text().strip() if api_key_path.exists() else os.getenv("OPENAI_API_KEY", "")
        return OpenAI(api_key=api_key), "openai"
    
    elif provider == "groq":
        api_key_path = API_KEY_DIR / "GROQ_API_KEY"
        api_key = api_key_path.read_text().strip() if api_key_path.exists() else os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Sign up at https://console.groq.com")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        ), "groq"
    
    elif provider == "nebius":
        api_key_path = API_KEY_DIR / "NEBIUS_API_KEY"
        api_key = api_key_path.read_text().strip() if api_key_path.exists() else os.getenv("NEBIUS_API_KEY", "")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY not set.")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.tokenfactory.nebius.com/v1/"
        ), "nebius"
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, groq, nebius")


def _build_rag_prompt(document: str, question: str) -> str:
    """Build the prompt for RAG layer evaluation."""
    return (
        f"You are a financial analyst.\n\n"
        f"FINANCIAL STATEMENT:\n{document}\n"
        f"QUESTION: {question}\n"
        f"RESPONSE: {{'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}\n"
    )


def _build_knowledge_prompt(question: str) -> str:
    """Build the prompt for Knowledge layer evaluation."""
    return (
        f"You are a financial analyst.\n\n"
        f"FINANCIAL STATEMENT:\n\n"
        f"QUESTION: {question}\n"
        f"RESPONSE: {{'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}\n"
    )


def _build_adversarial_prompt(metric: str, ground_truth: Any, fake_value: Any, question: str) -> str:
    """Build the prompt for Adversarial layer evaluation."""
    return (
        f"You are a financial analyst.\n\n"
        f"Source A: {metric} = {ground_truth}\n"
        f"Source B: {metric} = {fake_value}\n"
        f"Question: {question}\n\n"
        f"Which source is correct, Source A or Source B?\n"
        f"Do not invent a third option.\n\n"
        f"RESPONSE FORMAT (JSON only): "
        f"{{'source': 'A' or 'B' or 'UNKNOWN', 'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}\n\n"
    )


def prepare_jsonl(
    input_csv_path: str,
    output_jsonl_path: str,
    mapping_csv_path: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> int:
    """Reads the QA pairs CSV, builds JSONL batch requests for 3 layers and a mapping CSV.

    Returns the number of requests written.
    """
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} QA pairs from {input_csv_path}")
    print(f"Using provider: {provider}")

    # Ensure output directories exist
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Build lines and mapping
    lines: List[str] = []
    mapping_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        row_id = row.get("id", idx)
        question = row.get("generated_question", "")
        ground_truth = row.get("ground_truth_value", "N/A")
        entity = row.get("entity_name", "Unknown Company")
        year = row.get("year", "Unknown Year")
        metric = row.get("original_metric", "Unknown Metric")
        segment = row.get("segment", "Unknown Segment")

        # Generate document and fake value
        document = generate_document(segment, entity, year, metric, ground_truth)
        fake_value = modify_value(ground_truth, noise=0.15)

        # Create 3 requests per row: RAG, Knowledge, Adversarial
        layers = [
            ("rag", _build_rag_prompt(document, question)),
            ("knowledge", _build_knowledge_prompt(question)),
            ("adversarial", _build_adversarial_prompt(metric, ground_truth, fake_value, question)),
        ]

        for layer_name, prompt in layers:
            custom_id = f"eval_{row_id}_{layer_name}"

            body = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst. Output valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add response_format only for OpenAI (GROQ/NEBIUS may not support it in batch mode)
            if provider == "openai":
                body["response_format"] = {"type": "json_object"}

            one = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }

            lines.append(json.dumps(one))

        # Store mapping once per row
        mapping_rows.append(
            {
                "id": row_id,
                "question": question,
                "entity_name": entity,
                "year": year,
                "metric": metric,
                "segment": segment,
                "ground_truth_value": ground_truth,
                "fake_value": fake_value,
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


def submit_batch(input_jsonl_path: str, provider: str = "openai", completion_window: str = "24h") -> str:
    """Uploads the JSONL and creates a Batch job. Returns the batch id."""
    client, _ = get_client_and_provider(provider)
    
    with open(input_jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )

    print(f"Submitted batch: id={batch.id}, status={getattr(batch, 'status', 'unknown')}")
    return batch.id


def check_status(batch_id: str, provider: str = "openai") -> Dict[str, Any]:
    """Retrieves and prints batch status. Returns the batch dict."""
    client, _ = get_client_and_provider(provider)
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

    # Convert to plain dict
    try:
        batch_dict = batch.model_dump()
    except Exception:
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
    raw_csv_path: str,
    model: str,
    provider: str = "openai",
) -> int:
    """Downloads batch output JSONL, parses it, and writes the final results CSV.

    Returns number of parsed rows.
    """
    client, _ = get_client_and_provider(provider)
    batch = client.batches.retrieve(batch_id)
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("Batch output not available yet. Status: " + str(getattr(batch, "status", None)))

    # Download output content
    content_stream = client.files.content(output_file_id)
    try:
        data_bytes = content_stream.read()
    except AttributeError:
        data_text = getattr(content_stream, "text", None)
        if data_text is None:
            data_bytes = getattr(content_stream, "content", b"")
        else:
            data_bytes = data_text.encode("utf-8")

    data_text = data_bytes.decode("utf-8")

    # Save raw output JSONL
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        f.write(data_text)
    print(f"Saved batch output JSONL to {output_jsonl_path}")

    # Load mapping
    mapping_df = pd.read_csv(mapping_csv_path)
    mapping = {row["id"]: row for _, row in mapping_df.iterrows()}

    # Parse each line and organize by row_id and layer
    responses = {}  # row_id -> {layer -> response_data}
    for line in data_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)

        custom_id = obj.get("custom_id")
        error = obj.get("error")
        response = obj.get("response")

        # Parse custom_id: eval_{row_id}_{layer}
        parts = custom_id.split("_")
        if len(parts) < 3:
            continue
        row_id = int(parts[1])
        layer = parts[2]

        if row_id not in responses:
            responses[row_id] = {}

        if error:
            responses[row_id][layer] = {
                "raw": f"ERROR: {error}",
                "answer": None,
                "confidence": 0,
                "reasoning": f"ERROR: {error}",
            }
            continue

        if response:
            body = response.get("body", {})
            try:
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                responses[row_id][layer] = {
                    "raw": content,
                    "answer": parsed.get("answer"),
                    "confidence": parsed.get("confidence", 0),
                    "reasoning": parsed.get("reasoning"),
                }
            except Exception:
                responses[row_id][layer] = {
                    "raw": content if 'content' in locals() else "",
                    "answer": None,
                    "confidence": 0,
                    "reasoning": None,
                }

    # Build results and raw outputs
    results: List[Dict[str, Any]] = []
    raw_outputs: List[Dict[str, Any]] = []

    for row_id, layers_data in responses.items():
        base = mapping.get(row_id, {})
        
        rag_data = layers_data.get("rag", {})
        knowledge_data = layers_data.get("knowledge", {})
        adversarial_data = layers_data.get("adversarial", {})

        results.append(
            {
                "id": row_id,
                "model": model,
                "question": base.get("question"),
                "entity": base.get("entity_name"),
                "year": base.get("year"),
                "metric": base.get("metric"),
                "segment": base.get("segment"),
                "ground_truth": base.get("ground_truth_value"),
                "fake_value": base.get("fake_value"),
                "answer_rag": rag_data.get("answer"),
                "confidence_rag": rag_data.get("confidence", 0),
                "reasoning_rag": rag_data.get("reasoning"),
                "answer_knowledge": knowledge_data.get("answer"),
                "confidence_knowledge": knowledge_data.get("confidence", 0),
                "reasoning_knowledge": knowledge_data.get("reasoning"),
                "answer_adversarial": adversarial_data.get("answer"),
                "confidence_adversarial": adversarial_data.get("confidence", 0),
                "reasoning_adversarial": adversarial_data.get("reasoning"),
                "error": "",
            }
        )

        raw_outputs.append(
            {
                "id": row_id,
                "model": model,
                "raw_output_rag": rag_data.get("raw", ""),
                "raw_output_knowledge": knowledge_data.get("raw", ""),
                "raw_output_adversarial": adversarial_data.get("raw", ""),
                "error": "",
            }
        )

    # Write final CSVs
    Path(results_csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_csv_path, index=False)
    print(f"Wrote parsed results to {results_csv_path} ({len(results)} rows)")

    Path(raw_csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_outputs).to_csv(raw_csv_path, index=False)
    print(f"Wrote raw outputs to {raw_csv_path} ({len(raw_outputs)} rows)")

    return len(results)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--provider", default="openai", choices=["openai", "groq", "nebius"],
                        help="API provider to use")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=200)


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch runner for benchmark evaluation")
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
    p_submit.add_argument("--provider", default="openai", choices=["openai", "groq", "nebius"],
                          help="API provider to use")
    p_submit.add_argument("--window", default="24h", choices=["24h"])

    # status
    p_status = sub.add_parser("status", help="Check a batch job status")
    p_status.add_argument("batch_id")
    p_status.add_argument("--provider", default="openai", choices=["openai", "groq", "nebius"],
                          help="API provider to use")

    # collect
    p_collect = sub.add_parser("collect", help="Download output and write results CSV")
    p_collect.add_argument("batch_id")
    p_collect.add_argument("output_jsonl")
    p_collect.add_argument("mapping_csv")
    p_collect.add_argument("results_csv")
    p_collect.add_argument("raw_csv")
    p_collect.add_argument("--model", required=True, help="Model name used for the batch")
    p_collect.add_argument("--provider", default="openai", choices=["openai", "groq", "nebius"],
                           help="API provider to use")

    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare_jsonl(
            input_csv_path=args.input_csv,
            output_jsonl_path=args.output_jsonl,
            mapping_csv_path=args.mapping_csv,
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    elif args.cmd == "submit":
        submit_batch(args.input_jsonl, provider=args.provider, completion_window=args.window)
    elif args.cmd == "status":
        check_status(args.batch_id, provider=args.provider)
    elif args.cmd == "collect":
        collect_results(
            batch_id=args.batch_id,
            output_jsonl_path=args.output_jsonl,
            mapping_csv_path=args.mapping_csv,
            results_csv_path=args.results_csv,
            raw_csv_path=args.raw_csv,
            model=args.model,
            provider=args.provider,
        )


if __name__ == "__main__":
    main()
