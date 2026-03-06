#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e


echo "======================================================"
echo " Starting Quick Demo: FCA RAG Hallucination Benchmark "
echo "======================================================"
echo ""

echo "[1/3] Synthesising Ground Truth QA Pairs..."
bash scripts/create_qa_serial.sh configs/qa_serial/demo_run.yaml
echo "QA Pairs generated successfully."
echo ""

echo "[2/3] Evaluating LLMs RAG Accuracy against Ground Truth..."
bash scripts/run_serial.sh configs/llm_serial/gpt4o_demo_run.yaml
bash scripts/run_serial.sh configs/llm_serial/llama-3.1-8b-instant_demo_run.yaml
bash scripts/run_serial.sh configs/llm_serial/gemma-3n-E4B-it_demo_run.yaml
bash scripts/run_serial.sh configs/llm_serial/Qwen2.5-7B-Instruct-Turbo_demo_run.yaml
echo "LLM Evaluation complete."
echo ""

echo "[3/3] Analysing Results & Calculating Hallucination Rate..."
python scripts/analyze_rag_results.py \
  --input-path data/results/demo_run \
  --output-path data/rag_analysis/demo_run
echo ""

echo "=========================================================="
echo "Demo Complete! See the final RAG accuracy report above."
echo "=========================================================="
