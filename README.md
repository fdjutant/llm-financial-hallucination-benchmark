**Quantitative evaluation of LLM accuracy and hallucination on UK financial data extracted from regulatory XBRL filings.**

Financial accuracy is non-negotiable, yet LLMs struggle with numerical hallucinations. This benchmark targets that gap by rigorously testing models against real-world financial data. We provide an automated pipeline that extracts ground truth from FCA iXBRL filings to evaluate performance across RAG comprehension, parametric knowledge, and adversarial robustness.

## Architecture
```
iXBRL Filings (UK Financial Conduct Authority)
3 FTSE100 Companies (2022-2024)
      │
Fact Extraction & Cleaning
├── Bronze  : Raw Extraction
├── Silver  : Numeric Filtering
└── Gold    : Deduplication
      │
QA Pair Generation (Ground Truth)
1,562 QA Pairs
      │
LLM Benchmark Evaluation
├── 1. RAG Comprehension
├── 2. Parametric Knowledge
└── 3. Adversarial Robustness
      │
Results & Analysis
```

## Evaluation Methodology

### 3-Layer Benchmark Framework

**Layer 1: RAG (Retrieval-Augmented Generation)**
- **Question**: "What was Company X's 2023 revenue?"
- **Context**: Full financial statement excerpt provided
- **Metric**: Extraction accuracy (can the model read documents correctly?)

**Layer 2: Knowledge**
- **Question**: Same as Layer 1
- **Context**: No document provided
- **Metric**: Parametric knowledge (does the model "know" this fact?)

**Layer 3: Adversarial**
- **Question**: "Which source is correct: Source A (£1.2B) or Source B (£1.5B)?"
- **Context**: One correct, one fake value
- **Metric**: Source validation (can the model resist contradictions?)

## Results Summary

Based on evaluation of **1,562 QA pairs** across UK financials:

| Model                                |   N   | Correct | Incorrect | Accuracy (%) |
|--------------------------------------|-------|---------|-----------|--------------|
| google/gemma-3n-E4B-it               | 1562  |  1558   |     4     | 99.7         |
| gpt-4o                               | 1562  |  1432   |   130     | 91.7         |
| openai/gpt-oss-120b                  | 1562  |  1426   |   136     | 91.3         |
| openai/gpt-oss-20b                   | 1562  |  1388   |   174     | 88.9         |
| llama-3.1-8b-instant                 | 1562  |  1554   |     8     | 99.5         |
| llama-3.3-70b-versatile              | 1562  |  1527   |    35     | 97.8         |
| mistralai/Mistral-7B-Instruct-v0.3   | 1562  |  1514   |    48     | 96.9         |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2    | 1403  |   968   |   435     | 69.0         |
| Qwen/Qwen2.5-7B-Instruct-Turbo       | 1562  |  1538   |    24     | 98.5         |

**Key Findings:**
1. RAG accuracy for most models falls between 90% and 99%, highlighting strong but varied performance in document retrieval that can be used for automated processes
2. Models often introduce commas (,) into numeric values, such as 876,000,000 instead of 876000000. This can cause issues if the output is used directly in automated processes
3. In some incorrect cases, models provide additional context or formatting beyond the numeric value, such as "Negative 10,000,000.0" instead of -10000000 as specified in the financial statement. 

*See detailed RAG result for each model in `./data/rag_analysis/*.csv`*

## Features & Tech Stack
| Module      | Key Capabilities                                                                                       | Tech Stack                           |
| ----------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| Data Engine | Medallion Pipeline: Raw iXBRL (Bronze) → Numeric Filters (Silver) → Canonical Facts (Gold).            | arelle, pandas             |
| QA Gen      | LLM-Driven Synthesis: Generates 1,500+ question-answer pairs with reasoning derived from ground truth. | openai (Batch API), asyncio    |
| Benchmark   | 3-Layer Eval: RAG (Context), Parametric (Memory), and Adversarial (Robustness).                        | Custom metrics, multi-provider SDKs  |
| Operations  | Containerized Execution: Config-driven experiments, auto-resume, and secure API key management.        | Docker, yaml, jupyter, python-dotenv |
 
## Quick Setup (Recommended)

This project is configured with a **Dev Container** for a consistent, isolated environment.

### 1. Using VS Code or GitHub Codespaces
*   **GitHub Codespaces**: Click the **Code** button > **Codespaces** > **Create codespace on main**.
*   **VS Code (Local)**:
    1.  Install [Docker Desktop](https://www.docker.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
    2.  Open the project folder in VS Code
    3.  Click **"Reopen in Container"** when prompted (or run command `Dev Containers: Reopen in Container`)

### 2. Configure API Keys
The container expects API keys in a `.env` file (or environment variables). Add your keys:
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
NEBIUS_API_KEY=...
```

Alternatively, you can store your API keys in the `API_KEY` folder for better organization. Ensure the folder contains files named `OPENAI_API_KEY`, `GROQ_API_KEY`, and `NEBIUS_API_KEY` with the respective keys as their content. The project will automatically load these keys during runtime.

## Manual Installation (Optional)
If you prefer running locally without Docker:

1.  **Prerequisites**: Python 3.11+
2.  **Install**:
    ```bash
    git clone https://www.github.com/fdjutant/llm-financial-hallucination-benchmark/
    cd llm-financial-hallucination-benchmark
    
    # Conda (Recommended)
    conda env create -f environment.yml
    conda activate llmbenchmark
    ```
3.  **Verify**:
    ```bash
    python -m src.qa_generation.llm_qa_generator_batch --help
    ```

## Usage

### 0. Quick Demo Start
Run the full end-to-end pipeline on the demo dataset with a single command:
```bash
bash scripts/run_demo.sh
```
This executes three steps in sequence:
1. Generate QA pairs from `data/processed/demo/demo_gold.csv` 
2. Evaluate multiple LLMs against the QA pairs
3. Print the RAG accuracy table and write per-model CSVs to `data/rag_analysis/demo_run/`

### 1. Ground Truth Extraction
Pre-processed data is available in `data/processed/`. To regenerate from raw filings:
```python
# Notebook: notebooks/qa_data_pipelines.ipynb
from src.parsing.arelle_parser import process_html_files
from src.parsing.canonical_facts import create_silver_ground_truth, create_gold_ground_truth

# Bronze (Raw) -> Silver (Filter) -> Gold (Canonical)
bronze_df = process_html_files("data/raw/fca", "bronze.csv")
silver_df = create_silver_ground_truth(bronze_df, "silver.csv")
create_gold_ground_truth(silver_df, "gold.csv") 
```

### 2. Generate QA Pairs
Synthesize questions from ground truth using OpenAI's API.

**Option A: Batch (Recommended)**
Asynchronous — submits all requests at once and collects results after ~24h. Cost-effective for large datasets.
```bash
# Prepare & submit (default command: full)
bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml

# Check status, then collect once complete
bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml status
bash scripts/create_qa_batch.sh configs/qa_batch/gold_run.yaml collect
```

**Option B: Serial**
Synchronous — generates QA pairs row-by-row in real time. Simpler workflow with no wait, but slower and costlier for large datasets.
```bash
bash scripts/create_qa_serial.sh configs/qa_serial/demo_run.yaml
```

Config files live in `configs/qa_serial/` and `configs/qa_batch/`. Use the `demo_run.yaml` variants for a small subset.

### 3. Run Benchmark
Evaluate models using YAML configurations.

**Option A: Batch (Large Scale)**
Recommended for cost-effective evaluation of 1000+ pairs.
```bash
# Edit configs/experiments/gpt4o_benchmark.yaml
bash scripts/run_batch.sh configs/experiments/gpt4o_benchmark.yaml

# Collect results
python -m src.evaluation.benchmark_runner_batch --config ... collect <batch_id>
```

**Option B: Serial (Real-Time)**
Best for fast feedback.
```bash
# Edit configs/llm_serial/llama-3.1-8b-instant_demo_run.yaml
bash scripts/run_serial.sh configs/llm_serial/llama-3.1-8b-instant_demo_run.yaml
```

### 4. Analyze Results
Run the analysis script to print an accuracy summary table and export per-model RAG analysis CSVs.

```bash
# Serial demo run
python scripts/analyze_rag_results.py --input-path data/results/demo_run --output-path data/rag_analysis/demo_run

# Full batch results
python scripts/analyze_rag_results.py --input-path data/results/llm_batch --output-path data/rag_analysis
```

The script prints an aggregate table to stdout:
```
 model    N  correct  incorrect  accuracy (%)
 gpt-4o   6        5          1          83.3
```

Per-model RAG analysis CSVs (one per model) are written to the specified output folder.
Each file contains every result row enriched with `rag_outcome` (`correct` / `incorrect`).

For deeper exploration, open the notebook:
```bash
jupyter notebook notebooks/llm_benchmark_analysis.ipynb
```



## Acknowledgments

- **Arelle Project**: Open-source XBRL parsing library
- **Financial Conduct Authority**: UK regulatory filing data
- **OpenAI/Groq/Nebius**: LLM API providers