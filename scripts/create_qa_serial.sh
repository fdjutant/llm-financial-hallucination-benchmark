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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No configuration file provided${NC}"
    echo "Usage: bash scripts/create_qa_serial.sh <config_file>"
    echo "Example: bash scripts/create_qa_serial.sh configs/qa_serial/demo_run.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check for required API key
if [ -z "$OPENAI_API_KEY" ] && [ ! -f "/workspace/API_KEY/OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not found in environment or API_KEY/ directory${NC}"
fi

echo -e "${GREEN}Running:${NC} $CONFIG_FILE"

# Run the serial QA generator
python -m src.qa_generation.llm_qa_generator --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK]${NC} $CONFIG_FILE"
else
    echo -e "${RED}[FAIL]${NC} $CONFIG_FILE"
    exit 1
fi
