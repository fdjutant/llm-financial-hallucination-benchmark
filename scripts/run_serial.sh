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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No configuration file provided${NC}"
    echo "Usage: bash scripts/run_serial.sh <config_file>"
    echo "Example: bash scripts/run_serial.sh configs/serial/groq_model_by_model.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check for required environment variables
echo -e "${GREEN}Checking environment variables...${NC}"
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GROQ_API_KEY" ] && [ -z "$NEBIUS_API_KEY" ]; then
    echo -e "${YELLOW}Warning: No API keys found in environment variables${NC}"
    echo -e "${YELLOW}Falling back to API_KEY/ directory (not recommended for production)${NC}"
fi

# Print execution info
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Serial Benchmark Execution${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config File: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Execute the benchmark runner directly with config file
python -m src.evaluation.benchmark_runner_serial --config "$CONFIG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Execution completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Check results CSV files in the output directory"
    echo "2. Analyze results: jupyter notebook notebooks/llm_benchmark_analysis.ipynb"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Execution failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
