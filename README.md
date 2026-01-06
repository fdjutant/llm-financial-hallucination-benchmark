# Benchmarking Hallucinations on UK Financial Filings

This repository scaffolds a research pipeline that benchmarks hallucinations in LLMs on fact-based financial questions extracted from UK regulatory filings (Companies House iXBRL to start).

## Repository layout

```
src/
  data_download/
  parsing/
  qa_generation/
  evaluation/
data/
  raw/
  processed/
  qa/
  results/
notebooks/
```

The `src` tree contains Python modules (Python 3, pandas, standard library first) grouped by phase:

- `data_download/`: interactions with Companies House APIs/archives for UK filings plus storage manifests.
- `parsing/`: iXBRL loading, tag-to-fact mapping, and validation targeted at the UK taxonomy (FRC/IFRS).
- `qa_generation/`: template definitions and builders that emit programmatically verifiable QA pairs.
- `evaluation/`: LLM interface abstractions, graders, and analysis helpers for UK-specific benchmarks.

`data/` holds versioned artefacts (gitignored once we add `.gitignore`). `notebooks/` will host exploratory analysis/plots.

## Next steps

1. Decide on exact iXBRL parsing helper (e.g., `ixbrlparse`, `beautifulsoup4`, `lxml`).
2. Implement Companies House download client with cached manifests and FTSE company metadata table.
3. Define canonical fact schema + tag mapping for revenue, profit before tax, net income, total assets, and total liabilities within the UK taxonomy.
4. Author QA templates and grading heuristics, then add automated evaluation notebooks.


## Project Milestones (as of Dec 2025)

- Downloaded iXBRL filings for two UK companies (IDs: 08948140, 11270200) from Companies House.
- Successfully ingested and parsed iXBRL documents into pandas DataFrames using custom loaders in `/src/parsing`.
- Pipeline for data download and parsing is functional and reproducible for selected companies.
- Ready to proceed with fact extraction, Q&A generation, and LLM benchmarking.

Edge cases to track throughout:
- Missing or differently tagged facts between consolidated vs individual UK statements.
- Currency units, scale, and FY period handling for non-calendar ends.
- Inconsistent availability of iXBRL tags for older filings and subsidiaries.

# OpenAI Batch Runner for QA Generation

This repository includes a script to prepare, submit, monitor, and collect results from OpenAI batch jobs for generating financial QA datasets.

## Prerequisites

1. **API Key**: Ensure your OpenAI API key is stored in `API_KEY/OPENAI_API_KEY`.
2. **Dependencies**: Install required Python packages:
   ```bash
   pip install pandas openai
   ```

## Script Overview

The batch runner script is located at `src/qa_generation/openai_batch_runner.py`. It supports the following commands:

### 1. Prepare JSONL Requests

Builds JSONL batch requests and a mapping CSV from an input CSV file.

```bash
python src/qa_generation/openai_batch_runner.py prepare \
  data/processed/canonical_facts/bronze.csv \
  results/debug/batch/requests.jsonl \
  results/debug/batch/mapping.csv
```

- **Arguments**:
  - `input_csv`: Path to the input CSV file.
  - `output_jsonl`: Path to save the generated JSONL file.
  - `mapping_csv`: Path to save the mapping CSV file.
- **Optional Flags**:
  - `--model`: OpenAI model to use (default: `gpt-4o-mini`).
  - `--temperature`: Sampling temperature (default: `0.7`).
  - `--max_tokens`: Maximum tokens per response (default: `300`).

### 2. Submit Batch Job

Submits the prepared JSONL file as a batch job to OpenAI.

```bash
python src/qa_generation/openai_batch_runner.py submit \
  results/debug/batch/requests.jsonl \
  --window 24h
```

- **Arguments**:
  - `input_jsonl`: Path to the JSONL file.
- **Optional Flags**:
  - `--window`: Completion window (default: `24h`).

### 3. Check Batch Status

Checks the status of a submitted batch job.

```bash
python src/qa_generation/openai_batch_runner.py status <batch_id>
```

- **Arguments**:
  - `batch_id`: ID of the batch job.

### 4. Collect Results

Downloads the batch output, parses it, and writes the results to a CSV file.

```bash
python src/qa_generation/openai_batch_runner.py collect \
  <batch_id> \
  results/debug/batch/output.jsonl \
  results/debug/batch/mapping.csv \
  results/debug/batch/llama-3.3-70b-versatile_results.csv
```

- **Arguments**:
  - `batch_id`: ID of the batch job.
  - `output_jsonl`: Path to save the downloaded JSONL file.
  - `mapping_csv`: Path to the mapping CSV file.
  - `results_csv`: Path to save the final results CSV file.

## Example Workflow

1. **Prepare JSONL**:
   ```bash
   python src/qa_generation/openai_batch_runner.py prepare \
     data/processed/canonical_facts/bronze.csv \
     results/debug/batch/requests.jsonl \
     results/debug/batch/mapping.csv
   ```

2. **Submit Batch**:
   ```bash
   python src/qa_generation/openai_batch_runner.py submit \
     results/debug/batch/requests.jsonl
   ```

3. **Check Status**:
   ```bash
   python src/qa_generation/openai_batch_runner.py status <batch_id>
   ```

4. **Collect Results**:
   ```bash
   python src/qa_generation/openai_batch_runner.py collect \
     <batch_id> \
     results/debug/batch/output.jsonl \
     results/debug/batch/mapping.csv \
     results/debug/batch/llama-3.3-70b-versatile_results.csv
   ```

## Notes

- Ensure the input CSV follows the expected format with columns like `id`, `entity_name`, `year`, `canonical_fact_name`, `ground_truth_value`, and `segment`.
- Excluded segments: `Narrative_Disclosure`, `Other_Financial_Metric`.
- Results CSV includes columns for `generated_question`, `generated_answer`, and `generated_reasoning`.

---

# Batch Runner for Benchmark Evaluation

This repository includes a script to prepare, submit, monitor, and collect results from batch jobs for running benchmark evaluations on generated QA pairs. The script supports multiple API providers: **OpenAI**, **GROQ**, and **NEBIUS**.

## Prerequisites

1. **API Keys**: Ensure your API keys are stored in the appropriate files:
   - OpenAI: `API_KEY/OPENAI_API_KEY`
   - GROQ: `API_KEY/GROQ_API_KEY`
   - NEBIUS: `API_KEY/NEBIUS_API_KEY`
2. **Dependencies**: Install required Python packages:
   ```bash
   pip install pandas openai
   ```

## Script Overview

The benchmark batch runner script is located at `src/evaluation/openai_benchmark_batch_runner.py`. It supports the following commands:

### 1. Prepare JSONL Requests

Builds JSONL batch requests for 3 evaluation layers (RAG, Knowledge, Adversarial) and a mapping CSV from the QA pairs CSV.

```bash
python src/evaluation/openai_benchmark_batch_runner.py prepare \
  data/qa/qa_database/qa_pairs.csv \
  data/results/batch/requests.jsonl \
  data/results/batch/mapping.csv \
  --provider openai \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --max_tokens 200
```

- **Arguments**:
  - `input_csv`: Path to the QA pairs CSV file.
  - `output_jsonl`: Path to save the generated JSONL file.
  - `mapping_csv`: Path to save the mapping CSV file.
- **Optional Flags**:
  - `--provider`: API provider (`openai`, `groq`, `nebius`). Default: `openai`.
  - `--model`: Model to use. Default: `gpt-4o-mini`.
  - `--temperature`: Sampling temperature. Default: `0.0`.
  - `--max_tokens`: Maximum tokens per response. Default: `200`.

**Example models by provider:**
- OpenAI: `gpt-4o-mini`, `gpt-4o`
- GROQ: `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`
- NEBIUS: `deepseek-ai/DeepSeek-R1-0528`, `Qwen/Qwen2.5-72B-Instruct`

### 2. Submit Batch Job

Submits the prepared JSONL file as a batch job to the selected provider.

```bash
python src/evaluation/openai_benchmark_batch_runner.py submit \
  data/results/batch/requests.jsonl \
  --provider openai \
  --window 24h
```

- **Arguments**:
  - `input_jsonl`: Path to the JSONL file.
- **Optional Flags**:
  - `--provider`: API provider (`openai`, `groq`, `nebius`). Default: `openai`.
  - `--window`: Completion window. Default: `24h`.

### 3. Check Batch Status

Checks the status of a submitted batch job.

```bash
python src/evaluation/openai_benchmark_batch_runner.py status <batch_id> --provider openai
```

- **Arguments**:
  - `batch_id`: ID of the batch job.
- **Optional Flags**:
  - `--provider`: API provider (`openai`, `groq`, `nebius`). Default: `openai`.

### 4. Collect Results

Downloads the batch output, parses it, and writes the results and raw outputs to CSV files.

```bash
python src/evaluation/openai_benchmark_batch_runner.py collect \
  <batch_id> \
  data/results/batch/output.jsonl \
  data/results/batch/mapping.csv \
  data/results/batch/gpt-4o-mini_results.csv \
  data/results/batch/gpt-4o-mini_raw.csv \
  --model gpt-4o-mini \
  --provider openai
```

- **Arguments**:
  - `batch_id`: ID of the batch job.
  - `output_jsonl`: Path to save the downloaded JSONL file.
  - `mapping_csv`: Path to the mapping CSV file.
  - `results_csv`: Path to save the final results CSV file.
  - `raw_csv`: Path to save the raw outputs CSV file.
  - `--model`: Model name used for the batch (required).
- **Optional Flags**:
  - `--provider`: API provider (`openai`, `groq`, `nebius`). Default: `openai`.

## Example Workflow

### Using OpenAI

1. **Prepare JSONL**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py prepare \
     data/qa/qa_database/qa_pairs.csv \
     data/results/batch/openai_requests.jsonl \
     data/results/batch/openai_mapping.csv \
     --provider openai \
     --model gpt-4o-mini
   ```

2. **Submit Batch**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py submit \
     data/results/batch/openai_requests.jsonl \
     --provider openai
   ```

3. **Check Status**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py status <batch_id> --provider openai
   ```

4. **Collect Results**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py collect \
     <batch_id> \
     data/results/batch/openai_output.jsonl \
     data/results/batch/openai_mapping.csv \
     data/results/batch/gpt-4o-mini_results.csv \
     data/results/batch/gpt-4o-mini_raw.csv \
     --model gpt-4o-mini \
     --provider openai
   ```

### Using GROQ

1. **Prepare JSONL**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py prepare \
     data/qa/qa_database/qa_pairs.csv \
     data/results/batch/groq_requests.jsonl \
     data/results/batch/groq_mapping.csv \
     --provider groq \
     --model llama-3.3-70b-versatile
   ```

2. **Submit Batch**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py submit \
     data/results/batch/groq_requests.jsonl \
     --provider groq
   ```

3. **Check Status**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py status <batch_id> --provider groq
   ```

4. **Collect Results**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py collect \
     <batch_id> \
     data/results/batch/groq_output.jsonl \
     data/results/batch/groq_mapping.csv \
     data/results/batch/llama-3.3-70b_results.csv \
     data/results/batch/llama-3.3-70b_raw.csv \
     --model llama-3.3-70b-versatile \
     --provider groq
   ```

### Using NEBIUS

1. **Prepare JSONL**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py prepare \
     data/qa/qa_database/qa_pairs.csv \
     data/results/batch/nebius_requests.jsonl \
     data/results/batch/nebius_mapping.csv \
     --provider nebius \
     --model deepseek-ai/DeepSeek-R1-0528
   ```

2. **Submit Batch**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py submit \
     data/results/batch/nebius_requests.jsonl \
     --provider nebius
   ```

3. **Check Status**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py status <batch_id> --provider nebius
   ```

4. **Collect Results**:
   ```bash
   python src/evaluation/openai_benchmark_batch_runner.py collect \
     <batch_id> \
     data/results/batch/nebius_output.jsonl \
     data/results/batch/nebius_mapping.csv \
     data/results/batch/deepseek_results.csv \
     data/results/batch/deepseek_raw.csv \
     --model deepseek-ai/DeepSeek-R1-0528 \
     --provider nebius
   ```

## Notes

- Ensure the QA pairs CSV follows the expected format with columns like `id`, `generated_question`, `ground_truth_value`, `entity_name`, `year`, `original_metric`, and `segment`.
- The script generates 3 batch requests per QA pair (RAG, Knowledge, Adversarial layers).
- Results CSV includes evaluation outputs for all three layers with columns for answers, confidence scores, and reasoning.
- Raw outputs CSV preserves the complete JSON responses from the model for each layer.
- **Provider Compatibility**: All three providers (OpenAI, GROQ, NEBIUS) use OpenAI-compatible batch APIs with different base URLs and API keys.
