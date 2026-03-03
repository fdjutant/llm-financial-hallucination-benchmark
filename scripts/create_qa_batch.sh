#!/bin/bash
set -e

# ===================================================================
# QA Generation Batch Script
# ===================================================================
# This script automates QA generation workflow using batch APIs.
# Based on notebook: 20260103_create_qa_batch.ipynb
#
# Usage:
#   bash scripts/create_qa_batch.sh prepare
#   bash scripts/create_qa_batch.sh submit
#   bash scripts/create_qa_batch.sh status [batch_id]
#   bash scripts/create_qa_batch.sh collect [batch_id]
#   bash scripts/create_qa_batch.sh full
# ===================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths (can be overridden with environment variables)
INPUT_CSV="${QA_INPUT_CSV:-/workspace/data/processed/debug/tiny_sample_gold.csv}"
OUTPUT_JSONL="${QA_OUTPUT_JSONL:-/workspace/data/qa/debug/tiny/output.jsonl}"
MAPPING_CSV="${QA_MAPPING_CSV:-/workspace/data/qa/debug/tiny/mapping.csv}"
QA_PAIRS_CSV="${QA_PAIRS_CSV:-/workspace/data/qa/debug/tiny/qa_pairs.csv}"
# INPUT_CSV="${QA_INPUT_CSV:-/workspace/data/processed/canonical_facts/gold.csv}"
# OUTPUT_JSONL="${QA_OUTPUT_JSONL:-/workspace/data/qa/llm_batch/output.jsonl}"
# MAPPING_CSV="${QA_MAPPING_CSV:-/workspace/data/qa/llm_batch/mapping.csv}"
# QA_PAIRS_CSV="${QA_PAIRS_CSV:-/workspace/data/qa/llm_batch/qa_pairs.csv}"

# Model settings
MODEL="${QA_MODEL:-gpt-4o-mini}"
TEMPERATURE="${QA_TEMPERATURE:-0.7}"
MAX_TOKENS="${QA_MAX_TOKENS:-300}"
WINDOW="${QA_WINDOW:-24h}"

# Paths
PROJECT_ROOT="/workspace"
BATCH_ID_DIR="${PROJECT_ROOT}/BATCH_ID"

# Functions
save_batch_id() {
    local batch_id="$1"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local batch_id_file="${BATCH_ID_DIR}/${timestamp}_batchID"
    
    mkdir -p "${BATCH_ID_DIR}"
    echo "${batch_id}" > "${batch_id_file}"
    
    echo -e "${GREEN}✓ Batch ID saved to: ${batch_id_file}${NC}"
    echo "${batch_id_file}"
}

load_latest_batch_id() {
    if [ ! -d "${BATCH_ID_DIR}" ] || [ -z "$(ls -A ${BATCH_ID_DIR}/*_batchID 2>/dev/null)" ]; then
        echo -e "${RED}Error: No batch IDs found. Run 'submit' first.${NC}" >&2
        exit 1
    fi
    
    local latest_file=$(ls -t "${BATCH_ID_DIR}"/*_batchID | head -1)
    local batch_id=$(cat "${latest_file}")
    
    # Extract batch ID if it's in format "id=batch_xxx, status=..."
    if [[ "$batch_id" == *"id="* ]]; then
        batch_id=$(echo "$batch_id" | sed 's/.*id=\([^,]*\).*/\1/')
    fi
    
    echo -e "${BLUE}Using batch ID from: ${latest_file}${NC}" >&2
    echo "${batch_id}"
}

cmd_prepare() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Preparing LLM Batch Inputs${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Input CSV:     ${YELLOW}${INPUT_CSV}${NC}"
    echo -e "Output JSONL:  ${YELLOW}${OUTPUT_JSONL}${NC}"
    echo -e "Mapping CSV:   ${YELLOW}${MAPPING_CSV}${NC}"
    echo -e "Model:         ${YELLOW}${MODEL}${NC}"
    echo -e "Temperature:   ${YELLOW}${TEMPERATURE}${NC}"
    echo -e "Max Tokens:    ${YELLOW}${MAX_TOKENS}${NC}"
    echo ""
    
    python -m src.qa_generation.llm_qa_generator_batch prepare \
        "${INPUT_CSV}" \
        "${OUTPUT_JSONL}" \
        "${MAPPING_CSV}" \
        --model "${MODEL}" \
        --temperature "${TEMPERATURE}" \
        --max_tokens "${MAX_TOKENS}"
    
    echo ""
    echo -e "${GREEN}✓ Preparation complete!${NC}"
}

cmd_submit() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Submitting LLM Batch Job${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Input JSONL:   ${YELLOW}${OUTPUT_JSONL}${NC}"
    echo -e "Window:        ${YELLOW}${WINDOW}${NC}"
    echo ""
    
    # Capture the output which contains the batch ID
    local output=$(python -m src.qa_generation.llm_qa_generator_batch submit \
        "${OUTPUT_JSONL}" \
        --window "${WINDOW}" 2>&1)
    
    echo "$output"
    
    # Save the batch ID
    local batch_id_file=$(save_batch_id "$output")
    
    echo ""
    echo -e "${GREEN}✓ Batch submitted successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Check status: bash scripts/create_qa_batch.sh status"
    echo "2. Once complete, collect results: bash scripts/create_qa_batch.sh collect"
}

cmd_status() {
    local batch_id="$1"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Checking Batch Status${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    # Use provided batch_id or load latest
    if [ -z "$batch_id" ]; then
        batch_id=$(load_latest_batch_id)
    fi
    
    echo -e "Batch ID: ${YELLOW}${batch_id}${NC}"
    echo ""
    
    python -m src.qa_generation.llm_qa_generator_batch status "${batch_id}"
}

cmd_collect() {
    local batch_id="$1"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Collecting LLM Batch Results${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    # Use provided batch_id or load latest
    if [ -z "$batch_id" ]; then
        batch_id=$(load_latest_batch_id)
    fi
    
    echo -e "Batch ID:      ${YELLOW}${batch_id}${NC}"
    echo -e "Output JSONL:  ${YELLOW}${OUTPUT_JSONL}${NC}"
    echo -e "Mapping CSV:   ${YELLOW}${MAPPING_CSV}${NC}"
    echo -e "QA Pairs CSV:  ${YELLOW}${QA_PAIRS_CSV}${NC}"
    echo ""
    
    python -m src.qa_generation.llm_qa_generator_batch collect \
        "${batch_id}" \
        "${OUTPUT_JSONL}" \
        "${MAPPING_CSV}" \
        "${QA_PAIRS_CSV}"
    
    echo ""
    echo -e "${GREEN}✓ Results collected to: ${QA_PAIRS_CSV}${NC}"
}

cmd_full() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running Full QA Generation Workflow${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    # Step 1: Prepare
    cmd_prepare
    
    echo ""
    
    # Step 2: Submit
    cmd_submit
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Workflow Complete${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "The batch job has been submitted. Monitor its progress with:"
    echo "  bash scripts/create_qa_batch.sh status"
}

show_help() {
    echo "QA Generation Batch Script"
    echo ""
    echo "Usage: bash scripts/create_qa_batch.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  prepare              Prepare JSONL batch requests"
    echo "  submit               Submit batch job to OpenAI"
    echo "  status [batch_id]    Check batch job status (uses latest if not provided)"
    echo "  collect [batch_id]   Collect batch results (uses latest if not provided)"
    echo "  full                 Run full workflow (prepare + submit)"
    echo "  help                 Show this help message"
    echo ""
    echo "Environment Variables (optional):"
    echo "  QA_INPUT_CSV         Input CSV path (default: ${INPUT_CSV})"
    echo "  QA_OUTPUT_JSONL      Output JSONL path (default: ${OUTPUT_JSONL})"
    echo "  QA_MAPPING_CSV       Mapping CSV path (default: ${MAPPING_CSV})"
    echo "  QA_PAIRS_CSV         QA pairs CSV path (default: ${QA_PAIRS_CSV})"
    echo "  QA_MODEL             Model name (default: ${MODEL})"
    echo "  QA_TEMPERATURE       Temperature (default: ${TEMPERATURE})"
    echo "  QA_MAX_TOKENS        Max tokens (default: ${MAX_TOKENS})"
    echo "  QA_WINDOW            Completion window (default: ${WINDOW})"
    echo ""
    echo "Examples:"
    echo "  # Run full workflow with defaults"
    echo "  bash scripts/create_qa_batch.sh full"
    echo ""
    echo "  # Prepare with custom input"
    echo "  QA_INPUT_CSV=/path/to/custom.csv bash scripts/create_qa_batch.sh prepare"
    echo ""
    echo "  # Check status of specific batch"
    echo "  bash scripts/create_qa_batch.sh status batch_abc123"
    echo ""
    echo "  # Collect latest batch results"
    echo "  bash scripts/create_qa_batch.sh collect"
    echo ""
    echo "Required:"
    echo "  - OPENAI_API_KEY environment variable or API_KEY/OPENAI_API_KEY file"
    echo ""
}

# Main script
COMMAND="${1:-help}"

case "$COMMAND" in
    prepare)
        cmd_prepare
        ;;
    submit)
        cmd_submit
        ;;
    status)
        cmd_status "$2"
        ;;
    collect)
        cmd_collect "$2"
        ;;
    full)
        cmd_full
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
