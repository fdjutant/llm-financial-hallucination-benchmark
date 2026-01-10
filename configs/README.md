# Experiment Configurations

This directory contains YAML configuration files for benchmark experiments.

## Configuration Structure

Each YAML file defines:
- **experiment**: Metadata about the experiment
- **input**: Input data paths (QA pairs CSV)
- **output**: Output paths for all generated files
- **model**: LLM model settings (provider, model name, temperature, etc.)
- **batch**: Batch API settings

## Example Usage

```bash
# Run a benchmark experiment
bash scripts/run_batch.sh configs/experiments/gpt4o_chunk7_run.yaml
```

## Creating New Configurations

1. Copy an existing config file (e.g., `gpt4o_chunk7_run.yaml`)
2. Update the paths, model settings, and experiment metadata
3. Save with a descriptive name
4. Run using the shell script

## Environment Variables

API keys must be set as environment variables:
- `OPENAI_API_KEY` - for OpenAI models
- `GROQ_API_KEY` - for Groq models
- `NEBIUS_API_KEY` - for Nebius models

Never commit API keys to version control!
