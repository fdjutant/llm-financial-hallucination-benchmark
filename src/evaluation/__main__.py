import argparse
import asyncio
from .benchmark_runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Run LLM Benchmark")
    parser.add_argument("--qa_pairs", required=True, help="Path to QA pairs CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to test")
    parser.add_argument("--strategy", choices=["model_by_model", "row_by_row"], default="model_by_model", help="Execution strategy")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for checkpointing")
    parser.add_argument("--max_concurrency", type=int, default=None, help="Max concurrency for async execution")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        models_to_test=args.models,
        strategy=args.strategy,
        batch_size=args.batch_size,
        max_concurrency={"openai": args.max_concurrency} if args.max_concurrency else None,
    )

    asyncio.run(runner.run(args.qa_pairs, args.output_dir))

if __name__ == "__main__":
    main()
