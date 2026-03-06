#!/bin/bash
set -e

# ===================================================================
# Serial Benchmark Execution Script
# ===================================================================
# This script executes serial benchmark experiments using YAML configs
# Calls: src/evaluation/benchmark_runner_serial.py
# 
# Usage:
#   bash scripts/run_serial.sh <config_file>
#
# Arguments:
#   config_file: Path to YAML configuration file (e.g., configs/serial/groq_gpt-oss-120b.yaml)
#
# Examples:
#   bash scripts/run_serial.sh configs/serial/groq_model_by_model.yaml
#   bash scripts/run_serial.sh configs/serial/openai_row_by_row.yaml
# ===================================================================

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Error: No configuration file provided"
    echo "Usage: bash scripts/run_serial.sh <config_file>"
    echo "Example: bash scripts/run_serial.sh configs/serial/groq_model_by_model.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check for required environment variables
# if [ -z "$OPENAI_API_KEY" ] && [ -z "$GROQ_API_KEY" ] && [ -z "$NEBIUS_API_KEY" ]; then
#     echo "Warning: No API keys found in environment or API_KEY/ directory"
# fi

echo "Running: $CONFIG_FILE"

# Execute the benchmark runner directly with config file
python -m src.evaluation.benchmark_runner_serial --config "$CONFIG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "[OK] $CONFIG_FILE"
else
    echo "[FAIL] $CONFIG_FILE"
    exit 1
fi
