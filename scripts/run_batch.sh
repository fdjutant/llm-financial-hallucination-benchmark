#!/bin/bash
set -e

# ===================================================================
# Batch Benchmark Execution Script
# ===================================================================
# This script executes benchmark experiments using YAML configurations
# 
# Usage:
#   bash scripts/run_batch.sh <config_file> [command]
#
# Arguments:
#   config_file: Path to YAML configuration file (e.g., configs/experiments/gpt4o_chunk7_run.yaml)
#   command: Optional command to run (prepare, submit, full). Default: full
#
# Examples:
#   bash scripts/run_batch.sh configs/llm_batch/gpt4o_chunk7_run.yaml
#   bash scripts/run_batch.sh configs/llm_batch/gpt4o_chunk7_run.yaml prepare
#   bash scripts/run_batch.sh configs/llm_batch/gpt4o_chunk7_run.yaml submit
# ===================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE=""
COMMAND="full"

# Parse arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No configuration file provided${NC}"
    echo "Usage: bash scripts/run_batch.sh <config_file> [command]"
    echo "Example: bash scripts/run_batch.sh configs/experiments/gpt4o_chunk7_run.yaml"
    exit 1
fi

CONFIG_FILE=$1
COMMAND=${2:-full}

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Validate command
if [[ ! "$COMMAND" =~ ^(prepare|submit|full)$ ]]; then
    echo -e "${RED}Error: Invalid command '$COMMAND'. Must be one of: prepare, submit, full${NC}"
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
echo -e "${GREEN}Batch Benchmark Execution${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config File: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "Command: ${YELLOW}$COMMAND${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Execute the benchmark runner
python -m src.evaluation.benchmark_runner_batch --config "$CONFIG_FILE" "$COMMAND"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Execution completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    if [ "$COMMAND" = "full" ] || [ "$COMMAND" = "submit" ]; then
        echo ""
        echo -e "${YELLOW}Next Steps:${NC}"
        echo "1. Check batch status: python -m src.evaluation.benchmark_runner_batch status <batch_id> --provider <provider>"
        echo "2. Once complete, collect results: python -m src.evaluation.benchmark_runner_batch --config $CONFIG_FILE collect <batch_id>"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Execution failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
