# QA Batch Configuration Files

Each YAML file in this directory configures one QA generation run using the OpenAI Batch API.

## Format

```yaml
experiment:
  name: "<run_name>"
  description: "<description>"

input:
  gold_csv: "/path/to/gold.csv"       # Canonical gold facts CSV

output:
  base_dir: "/path/to/output/dir"
  requests_jsonl: "...requests.jsonl" # Batch API input requests
  mapping_csv: "...mapping.csv"       # ID-to-request mapping
  output_jsonl: "...output.jsonl"     # Raw batch API outputs
  qa_pairs_csv: "...qa_pairs.csv"     # Final parsed QA pairs

model:
  model_name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 300

batch:
  completion_window: "24h"
```

## Usage

```bash
bash scripts/create_qa_batch.sh configs/qa_batch/<config>.yaml [command]
```

Available commands: `prepare`, `submit`, `status`, `collect`, `full` (default).
