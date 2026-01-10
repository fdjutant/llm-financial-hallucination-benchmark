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
#   bash scripts/run_serial.sh configs/serial/groq_gpt-oss-120b.yaml
#   bash scripts/run_serial.sh configs/serial/llama-3.3-70b.yaml
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
    echo "Example: bash scripts/run_serial.sh configs/serial/groq_gpt-oss-120b.yaml"
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

# Parse YAML config and extract values using Python
echo "Parsing configuration..."

# Create a temporary Python script to parse the config
PARSE_OUTPUT=$(python3 -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    qa_pairs = config['input']['qa_pairs_csv']
    output_dir = config['output']['output_dir']
    models = ' '.join(config['models'])
    strategy = config['execution']['strategy']
    batch_size = config['execution']['batch_size']
    max_concurrency = config['execution'].get('max_concurrency', 'None')
    
    print(f'{qa_pairs}')
    print(f'{output_dir}')
    print(f'{models}')
    print(f'{strategy}')
    print(f'{batch_size}')
    print(f'{max_concurrency}')
except Exception as e:
    print(f'Error parsing config: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

# Check if parsing failed
if [ $? -ne 0 ]; then
    echo -e "${RED}$PARSE_OUTPUT${NC}"
    echo -e "${RED}Error: Failed to parse configuration file${NC}"
    exit 1
fi

# Read the output into variables
IFS=$'\n' read -d '' -r QA_PAIRS OUTPUT_DIR MODELS STRATEGY BATCH_SIZE MAX_CONCURRENCY <<< "$PARSE_OUTPUT" || true

# Check if parsing was successful
if [ -z "$QA_PAIRS" ]; then
    echo -e "${RED}Error: Failed to parse configuration file${NC}"
    exit 1
fi

# Display parsed configuration
echo -e "${GREEN}Configuration:${NC}"
echo -e "  QA Pairs:        ${YELLOW}${QA_PAIRS}${NC}"
echo -e "  Output Dir:      ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Models:          ${YELLOW}${MODELS}${NC}"
echo -e "  Strategy:        ${YELLOW}${STRATEGY}${NC}"
echo -e "  Batch Size:      ${YELLOW}${BATCH_SIZE}${NC}"
echo -e "  Max Concurrency: ${YELLOW}${MAX_CONCURRENCY}${NC}"
echo ""

# Validate input file exists
if [ ! -f "$QA_PAIRS" ]; then
    echo -e "${RED}Error: QA pairs file not found: $QA_PAIRS${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build the command
CMD="python -m src.evaluation.benchmark_runner_serial \
    --qa_pairs \"$QA_PAIRS\" \
    --output_dir \"$OUTPUT_DIR\" \
    --models $MODELS \
    --strategy \"$STRATEGY\" \
    --batch_size $BATCH_SIZE"

# Add max_concurrency if specified
if [ "$MAX_CONCURRENCY" != "None" ]; then
    CMD="$CMD --max_concurrency $MAX_CONCURRENCY"
fi

# Execute the benchmark
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Executing Benchmark${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Running: $CMD"
echo ""

eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Execution completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Results saved to:${NC}"
    echo "  $OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Check results CSV files in the output directory"
    echo "2. Analyze results: jupyter notebook notebooks/01_results_analysis.ipynb"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Execution failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
