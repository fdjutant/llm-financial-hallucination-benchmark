#!/bin/bash
set -e

# ===================================================================
# QA Generation Batch Script
# ===================================================================
# This script automates QA generation workflow using YAML configurations
# and the OpenAI Batch API.
#
# Usage:
#   bash scripts/create_qa_batch.sh <config_file> [command]
#
# Arguments:
#   config_file: Path to YAML configuration file (e.g., configs/qa_batch/demo_run.yaml)
#   command: Optional command to run (prepare, submit, full, status, collect). Default: full
#
# Examples:
#   bash scripts/create_qa_batch.sh configs/qa_batch/demo_run.yaml
#   bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml prepare
#   bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml submit
#   bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml status
#   bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml status batch_abc123
#   bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml collect
#   bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml collect batch_abc123
# ===================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
PROJECT_ROOT="/workspace"
BATCH_ID_DIR="${PROJECT_ROOT}/BATCH_ID"

# Default values
CONFIG_FILE=""
COMMAND="full"

# Parse arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No configuration file provided${NC}"
    echo "Usage: bash scripts/create_qa_batch.sh <config_file> [command]"
    echo "Example: bash scripts/create_qa_batch.sh configs/qa_batch/demo_run.yaml"
    exit 1
fi

CONFIG_FILE=$1
COMMAND=${2:-full}
BATCH_ID_ARG=${3:-}

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Validate command
if [[ ! "$COMMAND" =~ ^(prepare|submit|full|status|collect)$ ]]; then
    echo -e "${RED}Error: Invalid command '$COMMAND'. Must be one of: prepare, submit, full, status, collect${NC}"
    exit 1
fi

# Check for required API key
echo -e "${GREEN}Checking environment variables...${NC}"
if [ -z "$OPENAI_API_KEY" ] && [ ! -f "${PROJECT_ROOT}/API_KEY/OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not found in environment or API_KEY/ directory${NC}"
fi

# Helper: save batch ID returned by submit
save_batch_id() {
    local batch_id="$1"
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local batch_id_file="${BATCH_ID_DIR}/${timestamp}_batchID"

    mkdir -p "${BATCH_ID_DIR}"
    echo "${batch_id}" > "${batch_id_file}"

    echo -e "${GREEN}✓ Batch ID saved to: ${batch_id_file}${NC}"
}

# Helper: load the most recent batch ID from BATCH_ID/
load_latest_batch_id() {
    if [ ! -d "${BATCH_ID_DIR}" ] || [ -z "$(ls -A "${BATCH_ID_DIR}"/*_batchID 2>/dev/null)" ]; then
        echo -e "${RED}Error: No batch IDs found. Run 'submit' first.${NC}" >&2
        exit 1
    fi

    local latest_file
    latest_file=$(ls -t "${BATCH_ID_DIR}"/*_batchID | head -1)
    local batch_id
    batch_id=$(cat "${latest_file}")

    # Extract batch ID if stored as "id=batch_xxx, status=..."
    if [[ "$batch_id" == *"id="* ]]; then
        batch_id=$(echo "$batch_id" | sed 's/.*id=\([^,]*\).*/\1/')
    fi

    echo -e "${BLUE}Using batch ID from: ${latest_file}${NC}" >&2
    echo "${batch_id}"
}

# Print execution info
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}QA Generation Batch Execution${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config File: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "Command:     ${YELLOW}$COMMAND${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Execute workflow
if [ "$COMMAND" = "prepare" ]; then
    python -m src.qa_generation.llm_qa_generator_batch --config "$CONFIG_FILE" prepare

elif [[ "$COMMAND" =~ ^(submit|full)$ ]]; then
    # Run prepare first when doing a full run
    if [ "$COMMAND" = "full" ]; then
        python -m src.qa_generation.llm_qa_generator_batch --config "$CONFIG_FILE" prepare
        echo ""
    fi

    # Capture submit output to extract and persist the batch ID
    SUBMIT_OUTPUT=$(python -m src.qa_generation.llm_qa_generator_batch --config "$CONFIG_FILE" submit 2>&1)
    echo "$SUBMIT_OUTPUT"

    BATCH_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP "batch_[A-Za-z0-9]+" | head -1 || true)
    if [ -n "$BATCH_ID" ]; then
        save_batch_id "$BATCH_ID"
    else
        echo -e "${YELLOW}Warning: Could not extract batch ID from submit output${NC}"
    fi

elif [ "$COMMAND" = "status" ]; then
    BATCH_ID="${BATCH_ID_ARG}"
    if [ -z "$BATCH_ID" ]; then
        BATCH_ID=$(load_latest_batch_id)
    fi
    echo -e "Batch ID: ${YELLOW}${BATCH_ID}${NC}"
    echo ""
    python -m src.qa_generation.llm_qa_generator_batch status "${BATCH_ID}"

elif [ "$COMMAND" = "collect" ]; then
    BATCH_ID="${BATCH_ID_ARG}"
    if [ -z "$BATCH_ID" ]; then
        BATCH_ID=$(load_latest_batch_id)
    fi
    echo -e "Batch ID: ${YELLOW}${BATCH_ID}${NC}"
    echo ""
    python -m src.qa_generation.llm_qa_generator_batch --config "$CONFIG_FILE" collect "${BATCH_ID}"
fi

# Final status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Execution completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"

    if [[ "$COMMAND" =~ ^(submit|full)$ ]]; then
        echo ""
        echo -e "${YELLOW}Next Steps:${NC}"
        echo "1. Check batch status:"
        echo "   bash scripts/create_qa_batch.sh $CONFIG_FILE status"
        echo "2. Once complete, collect results:"
        echo "   bash scripts/create_qa_batch.sh $CONFIG_FILE collect"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Execution failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
