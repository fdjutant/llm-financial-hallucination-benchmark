#!/bin/bash
# Helper script to create a new experiment configuration

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Experiment Config Generator ===${NC}"
echo ""

# Get experiment name
read -p "Experiment name (e.g., gpt4o_chunk1): " exp_name
if [ -z "$exp_name" ]; then
    echo -e "${RED}Error: Experiment name is required${NC}"
    exit 1
fi

# Get provider
echo ""
echo "Select provider:"
echo "1) openai"
echo "2) groq"
echo "3) nebius"
read -p "Enter choice [1-3]: " provider_choice

case $provider_choice in
    1) provider="openai";;
    2) provider="groq";;
    3) provider="nebius";;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Get model name
read -p "Model name (e.g., gpt-4o, llama-3.3-70b-versatile): " model_name
if [ -z "$model_name" ]; then
    echo -e "${RED}Error: Model name is required${NC}"
    exit 1
fi

# Get input file
read -p "Input CSV path (e.g., /workspace/data/qa/llm_batch/chunks/qa_pairs_chunk_1.csv): " input_csv
if [ -z "$input_csv" ]; then
    echo -e "${RED}Error: Input CSV path is required${NC}"
    exit 1
fi

# Get output directory
read -p "Output directory (e.g., /workspace/data/results/llm_batch/${model_name}/chunk_1): " output_dir
if [ -z "$output_dir" ]; then
    echo -e "${RED}Error: Output directory is required${NC}"
    exit 1
fi

# Get temperature
read -p "Temperature [0.0]: " temperature
temperature=${temperature:-0.0}

# Get max tokens
read -p "Max tokens [200]: " max_tokens
max_tokens=${max_tokens:-200}

# Create config file
config_file="configs/experiments/${exp_name}.yaml"

echo ""
echo -e "${YELLOW}Creating config file: ${config_file}${NC}"

cat > "$config_file" <<EOF
# Configuration for ${exp_name}
# Generated: $(date +"%Y-%m-%d %H:%M:%S")

experiment:
  name: "${exp_name}"
  description: "Benchmark ${model_name} on QA pairs"

input:
  qa_pairs_csv: "${input_csv}"

output:
  base_dir: "${output_dir}"
  requests_jsonl: "${output_dir}/requests.jsonl"
  mapping_csv: "${output_dir}/mapping.csv"
  output_jsonl: "${output_dir}/output.jsonl"
  results_csv: "${output_dir}_results.csv"
  raw_csv: "${output_dir}_raw.csv"

model:
  provider: "${provider}"
  model_name: "${model_name}"
  temperature: ${temperature}
  max_tokens: ${max_tokens}

batch:
  completion_window: "24h"

# Environment variable required: ${provider^^}_API_KEY
EOF

echo -e "${GREEN}✓ Config file created successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review and edit the config if needed:"
echo "   ${config_file}"
echo ""
echo "2. Set your API key as environment variable:"
echo "   export ${provider^^}_API_KEY='your-key-here'"
echo ""
echo "3. Run the experiment:"
echo "   bash scripts/run_batch.sh ${config_file}"
echo ""
