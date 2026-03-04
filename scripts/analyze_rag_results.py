#!/usr/bin/env python
"""
Analyze benchmark results and generate per-model RAG analysis CSVs.

Usage:
    python scripts/analyze_rag_results.py --input-path <results_folder> --output-path <rag_output_folder>

Examples:
    python scripts/analyze_rag_results.py --input-path data/results/llm_serial/demo_run --output-path data/rag_analysis/demo_run
    python scripts/analyze_rag_results.py --input-path data/results/llm_batch --output-path data/rag_analysis
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.analysis import analyze_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and export per-model RAG analysis CSVs."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to folder containing model result subfolders (each with *_results.csv).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output folder where per-model RAG analysis CSVs will be written.",
    )
    args = parser.parse_args()

    results_folder = str(Path(args.input_path).resolve())
    rag_output_folder = str(Path(args.output_path).resolve())

    print(f"Results folder:    {results_folder}")
    print(f"RAG output folder: {rag_output_folder}")
    print()

    analyze_results(
        base_folder=results_folder,
        rag_output_folder=rag_output_folder,
    )


if __name__ == "__main__":
    main()
