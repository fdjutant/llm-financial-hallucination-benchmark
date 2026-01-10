import os
import pandas as pd
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import asyncio
from .llm_interface import (
    call_llm_with_prompt,
    generate_document,
    modify_value,
    robust_extract_json,
)

class BenchmarkRunner:
    def __init__(self, models_to_test, strategy="model_by_model", batch_size=25, max_concurrency=None):
        """
        Initialize the BenchmarkRunner.

        Args:
            models_to_test (list[str]): List of models to evaluate.
            strategy (str): Execution strategy ('model_by_model' or 'row_by_row').
            batch_size (int): Number of rows to process before saving progress.
            max_concurrency (dict): Max concurrency per provider (e.g., {'openai': 5, 'groq': 3}).
        """
        self.models_to_test = models_to_test
        self.strategy = strategy
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency or {}

    async def run(self, qa_pairs_path, output_dir):
        """
        Run the benchmark based on the selected strategy.

        Args:
            qa_pairs_path (str): Path to the QA pairs CSV file.
            output_dir (str): Directory to save results.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(qa_pairs_path)
        print(f"Loaded {len(df)} QA pairs")

        if self.strategy == "model_by_model":
            for model in self.models_to_test:
                await self._evaluate_model(df, model, output_dir)
        elif self.strategy == "row_by_row":
            for idx, row in df.iterrows():
                await self._evaluate_row(row, output_dir)

    async def _evaluate_model(self, df, model, output_dir):
        """
        Evaluate all rows for a single model.

        Args:
            df (pd.DataFrame): DataFrame containing QA pairs.
            model (str): Model name to evaluate.
            output_dir (str): Directory to save results.
        """
        results_path = Path(output_dir) / f"{model.replace('/', '_')}_results.csv"
        raw_path = Path(output_dir) / f"{model.replace('/', '_')}_raw.csv"

        # Resume from checkpoint if files exist
        processed_ids = set()
        if results_path.exists():
            processed_ids = set(pd.read_csv(results_path)["id"])
            print(f"Resuming {model}: {len(processed_ids)} rows already processed")

        results = []
        raw_outputs = []
        semaphore = asyncio.Semaphore(self.max_concurrency.get("openai", 1))

        for idx, row in df.iterrows():
            if row["id"] in processed_ids:
                continue

            async with semaphore:
                try:
                    result, raw_output = await self._evaluate_row(row, model)
                    results.append(result)
                    raw_outputs.append(raw_output)
                except Exception as e:
                    print(f"Error processing row {row['id']} for model {model}: {e}")
                    results.append(self._safe_result(row, model, error=str(e)))
                    raw_outputs.append(self._safe_raw_output(row, model, error=str(e)))

            # Save progress every batch_size rows
            if (idx + 1) % self.batch_size == 0:
                self._save_results(results, results_path)
                self._save_results(raw_outputs, raw_path)
                results.clear()
                raw_outputs.clear()

        # Final save
        self._save_results(results, results_path)
        self._save_results(raw_outputs, raw_path)

    async def _evaluate_row(self, row, model):
        """
        Evaluate a single row for a given model.

        Args:
            row (pd.Series): Row from the QA pairs DataFrame.
            model (str): Model name to evaluate.

        Returns:
            tuple: (result dict, raw_output dict)
        """
        question = row["generated_question"]
        ground_truth = row["ground_truth_value"]
        entity = row["entity_name"]
        year = row["year"]
        metric = row["original_metric"]
        segment = row["segment"]

        document = generate_document(segment, entity, year, metric, ground_truth)

        # Layer 1: RAG
        prompt_rag = self._build_prompt(document, question)
        llm_output_rag = await self._call_with_retry(model, prompt_rag)
        answer_rag, confidence_rag, reasoning_rag = robust_extract_json(llm_output_rag.full_answer)

        # Layer 2: Knowledge
        prompt_knowledge = self._build_prompt("", question)
        llm_output_knowledge = await self._call_with_retry(model, prompt_knowledge)
        answer_knowledge, confidence_knowledge, reasoning_knowledge = robust_extract_json(llm_output_knowledge.full_answer)

        # Layer 3: Adversarial
        fake_value = modify_value(ground_truth, noise=0.15)
        prompt_adversarial = self._build_adversarial_prompt(metric, ground_truth, fake_value, question)
        llm_output_adversarial = await self._call_with_retry(model, prompt_adversarial)
        answer_adversarial, confidence_adversarial, reasoning_adversarial = robust_extract_json(llm_output_adversarial.full_answer)
        
        result = {
            "id": row["id"],
            "model": model,
            "question": question,
            "entity": entity,
            "year": year,
            "metric": metric,
            "segment": segment,
            "ground_truth": ground_truth,
            "fake_value": fake_value,
            "answer_rag": answer_rag,
            "confidence_rag": confidence_rag,
            "reasoning_rag": reasoning_rag,
            "answer_knowledge": answer_knowledge,
            "confidence_knowledge": confidence_knowledge,
            "reasoning_knowledge": reasoning_knowledge,
            "answer_adversarial": answer_adversarial,
            "confidence_adversarial": confidence_adversarial,
            "reasoning_adversarial": reasoning_adversarial,
            "error": "",
        }

        raw_output = {
            "id": row["id"],
            "model": model,
            "raw_output_rag": llm_output_rag.full_answer,
            "raw_output_knowledge": llm_output_knowledge.full_answer,
            "raw_output_adversarial": llm_output_adversarial.full_answer,
            "error": "",
        }

        return result, raw_output

    async def _call_with_retry(self, model, prompt, retries=2):
        """
        Call the LLM with retries on failure.

        Args:
            model (str): Model name.
            prompt (str): Prompt to send to the model.
            retries (int): Number of retries on failure.

        Returns:
            LLMOutput: Output from the LLM.
        """
        for attempt in range(retries + 1):
            try:
                return call_llm_with_prompt(model, prompt)
            except Exception as e:
                if attempt == retries:
                    raise e
                await asyncio.sleep(2 ** attempt)

    def _save_results(self, results, path):
        """
        Save results to a CSV file.

        Args:
            results (list[dict]): List of result dictionaries.
            path (str): Path to the output file.
        """
        df = pd.DataFrame(results)
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def _safe_result(self, row, model, error=""):
        """
        Generate a safe result for a failed row.

        Args:
            row (pd.Series): Row from the QA pairs DataFrame.
            model (str): Model name.
            error (str): Error message.

        Returns:
            dict: Safe result dictionary.
        """
        return {
            "id": row["id"],
            "model": model,
            "question": row["generated_question"],
            "entity": row["entity_name"],
            "year": row["year"],
            "metric": row["original_metric"],
            "segment": row["segment"],
            "ground_truth": row["ground_truth_value"],
            "fake_value": "",
            "answer_rag": "",
            "confidence_rag": 0,
            "answer_knowledge": "",
            "confidence_knowledge": 0,
            "answer_adversarial": "",
            "confidence_adversarial": 0,
            "rag_correct": False,
            "knowledge_correct": False,
            "hallucinated": False,
            "adversarial_correct": False,
            "error": error,
        }

    def _safe_raw_output(self, row, model, error=""):
        """
        Generate a safe raw output for a failed row.

        Args:
            row (pd.Series): Row from the QA pairs DataFrame.
            model (str): Model name.
            error (str): Error message.

        Returns:
            dict: Safe raw output dictionary.
        """
        return {
            "id": row["id"],
            "model": model,
            "raw_output_rag": "",
            "raw_output_knowledge": "",
            "raw_output_adversarial": "",
            "error": error,
        }

    def _build_prompt(self, document, question):
        """
        Build the prompt for the RAG and Knowledge layers.

        Args:
            document (str): Financial document text.
            question (str): Question to ask the model.

        Returns:
            str: Prompt string.
        """
        return (
            f"You are a financial analyst.\n\n"
            f"FINANCIAL STATEMENT:\n{document}\n"
            f"QUESTION: {question}\n"
            f"RESPONSE: {{'answer': '...', 'confidence': 0-100, 'reasoning': '...'}}\n"
        )

    def _build_adversarial_prompt(self, metric, ground_truth, fake_value, question):
        """
        Build the prompt for the Adversarial layer.

        Args:
            metric (str): Metric name.
            ground_truth (str): Ground truth value.
            fake_value (str): Fake value to test adversarial robustness.
            question (str): Question to ask the model.

        Returns:
            str: Prompt string.
        """
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Run LLM Benchmark",
                epilog="Example usage:\n"
                    "  python -m ./src/evaluation --qa_pairs ./src/data/qa/debug/qa_pairs.csv "
                    "--output_dir results --models openai/gpt-oss-120b llama-3.3-70b-versatile "
                    "--strategy row_by_row "
                    "--batch_size 50 --max_concurrency 10")
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