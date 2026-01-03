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
