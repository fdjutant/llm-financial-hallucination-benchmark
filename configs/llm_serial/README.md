# Serial Benchmark Configurations

This directory contains YAML configuration files for serial (synchronous) benchmark experiments.

## Configuration Structure

Each YAML file defines:
- **experiment**: Metadata about the experiment
- **input**: Input data paths (QA pairs CSV)
- **output**: Output directory for results
- **model**: Model configuration (provider, temperature, max_tokens)
- **models**: List of model names to evaluate
- **execution**: Execution settings (strategy, batch size, concurrency)

## Example Configuration

```yaml
experiment:
  name: "my_serial_benchmark"
  description: "Description of this benchmark run"

input:
  qa_pairs_csv: "/workspace/data/qa/llm_batch/qa_pairs.csv"

output:
  output_dir: "/workspace/data/results/serial/my_model"

model:
  provider: "openai"  # Options: openai, groq, nebius
  temperature: 0.0  # Optional: omit or set to null for models that don't support it
  max_tokens: 200  # Maximum tokens in response

models:
  # List one or more models
  - "openai/gpt-oss-120b"
  - "llama-3.3-70b-versatile"

execution:
  # Strategy: 'model_by_model' or 'row_by_row'
  strategy: "model_by_model"
  
  # Save checkpoint every N rows
  batch_size: 25
  
  # Max concurrent async requests
  max_concurrency: 5
```

## Usage

```bash
bash scripts/run_serial.sh configs/serial/your_config.yaml
```

## Execution Strategies

### model_by_model (Recommended)
- Completes all rows for one model before moving to next
- Best for multi-model comparisons
- Easier to resume if interrupted
- Example: Process all 1000 rows for GPT-4, then all 1000 for Llama

### row_by_row
- Processes one row across all models before next row
- Best for quick sampling
- More balanced progress across models
- Example: Row 1 on GPT-4 + Llama, then Row 2 on GPT-4 + Llama

## Checkpointing

The serial runner automatically saves progress:
- Checkpoints created every `batch_size` rows
- Results saved to CSV incrementally
- On restart, automatically resumes from last checkpoint
- No manual intervention required

## Model Names

Use the exact model identifiers expected by the LLM API:

**OpenAI:**
- `gpt-4o`
- `gpt-4o-mini-search-preview` (Note: does not support temperature parameter)
- `gpt-3.5-turbo`

**GROQ:**
- `llama-3.3-70b-versatile`
- `openai/gpt-oss-120b`
- `mixtral-8x7b-32768`

**Nebius:**
- `deepseek-ai/DeepSeek-R1-0528`
- `Qwen/Qwen2.5-72B-Instruct`

## Model Parameters

### Temperature
- Controls randomness in model output (0.0 = deterministic, 2.0 = creative)
- **Optional**: Some models (e.g., `gpt-4o-mini-search-preview`) don't support it
- To omit: Remove the `temperature` line or set to `null`
- Default: 0.0 (when supported)

### Max Tokens
- Maximum number of tokens in the model's response
- **Required**: Always specify this parameter
- Recommended: 200 for financial QA (our use case)
- Increase if you need longer responses (e.g., 1000 for detailed analysis)

**Example for models without temperature support:**
```yaml
model:
  provider: "openai"
  # temperature parameter omitted
  max_tokens: 1000
```

## Concurrency Settings

`max_concurrency` controls async request limit:
- **Low (1-3)**: Conservative, fewer rate limit issues
- **Medium (5-10)**: Balanced speed and reliability
- **High (15+)**: Fast but may hit rate limits

Recommended starting points:
- OpenAI: 5-10
- GROQ: 10-20
- Nebius: 5-10

Adjust based on your API tier and rate limits.

## Environment Variables

Required API keys (set as environment variables):
```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
export NEBIUS_API_KEY="..."
```

Or use file-based fallback in `API_KEY/` directory.

## Output Structure

Results are saved to the specified output directory:

```
output_dir/
├── model_name_results.csv    # Parsed results with evaluation metrics
│   ├── id                     # Row ID
│   ├── question               # Question text
│   ├── ground_truth           # Correct answer
│   ├── answer_rag             # RAG layer answer
│   ├── confidence_rag         # RAG confidence score
│   ├── rag_correct            # Boolean correctness flag
│   ├── answer_knowledge       # Knowledge layer answer
│   ├── ... (similar for other layers)
│
└── model_name_raw.csv         # Raw JSON outputs from model
    ├── id
    ├── model
    ├── raw_output_rag         # Full JSON response
    ├── raw_output_knowledge
    └── raw_output_adversarial
```

## Creating New Configurations

1. Copy an existing config:
   ```bash
   cp configs/serial/groq_gpt-oss-120b.yaml configs/serial/my_new_config.yaml
   ```

2. Edit the configuration:
   ```bash
   nano configs/serial/my_new_config.yaml
   ```

3. Update:
   - Experiment name and description
   - Model names
   - Output directory
   - Execution parameters

4. Run:
   ```bash
   bash scripts/run_serial.sh configs/serial/my_new_config.yaml
   ```

## Troubleshooting

### "Module not found" errors
Ensure you're running from project root:
```bash
cd /workspace
bash scripts/run_serial.sh configs/serial/your_config.yaml
```

### Rate limit errors
Reduce `max_concurrency` in the config file.

### Out of memory
Reduce `batch_size` to save progress more frequently.

### Resume from checkpoint
Just run the same command again - the script automatically detects existing results and resumes.

## Related Documentation

- [Main README](../../README.md) - Project overview
- [Scripts README](../../scripts/README.md) - Script documentation
- [Architecture](../../ARCHITECTURE.md) - System design
