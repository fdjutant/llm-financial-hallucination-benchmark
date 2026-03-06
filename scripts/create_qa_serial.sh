#!/bin/bash
set -e

# ===================================================================
# Serial QA Generation Script
# ===================================================================
# This script generates QA pairs row-by-row without the Batch API.
# Calls: src/qa_generation/llm_qa_generator.py
#
# Usage:
#   bash scripts/create_qa_serial.sh <config_file>
#
# Arguments:
#   config_file: Path to YAML configuration file (e.g., configs/qa_serial/demo_run.yaml)
#
# Examples:
#   bash scripts/create_qa_serial.sh configs/qa_serial/demo_run.yaml
#   bash scripts/create_qa_serial.sh configs/qa_serial/gold_run.yaml
# ===================================================================

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Error: No configuration file provided"
    echo "Usage: bash scripts/create_qa_serial.sh <config_file>"
    echo "Example: bash scripts/create_qa_serial.sh configs/qa_serial/demo_run.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check for required API key
if [ -z "$OPENAI_API_KEY" ] && [ ! -f "/workspace/API_KEY/OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not found in environment or API_KEY/ directory"
fi

echo "Running: $CONFIG_FILE"

# Run the serial QA generator
python -m src.qa_generation.llm_qa_generator --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "[OK] $CONFIG_FILE"
else
    echo "[FAIL] $CONFIG_FILE"
    exit 1
fi
