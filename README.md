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

| Model | RAG Accuracy (%) | RAG Hallucination (%) | Notes |
|-------|---------|-----------------|-------|
| gpt-4o | 91.7 | 8.3 | Occasional drift from context; tends to overwrite facts with internal knowledge |
| gpt-oss-120b | 91.3 | 8.7 | High error rate; frequently hallucinates despite correct context |
| llama-3.1-8b-instant | 99.5 | 0.5 | Exceptional faithfulness to provided financial documents |

**Key Findings:**
1. RAG performance varies significantly (90-99%)
2. All models struggle with parametric knowledge (0-0.3% accuracy without documents)
3. Adversarial robustness is uniformly low (0-5%)

*See llm_benchmark_analysis.ipynb for detailed analysis.*

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
    conda activate mlenv
    ```
3.  **Verify**:
    ```bash
    python -m src.qa_generation.llm_qa_generator_batch --help
    ```

## Usage

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

### 2. Generate QA Pairs (Batch API)
Synthesize questions from ground truth using OpenAI's Batch API.
```bash
# 1. Prepare & Submit
python -m src.qa_generation.llm_qa_generator_batch prepare gold.csv requests.jsonl mapping.csv --model gpt-4o-mini
python -m src.qa_generation.llm_qa_generator_batch submit requests.jsonl

# 2. Track & Collect (after 24h)
python -m src.qa_generation.llm_qa_generator_batch status <batch_id>
python -m src.qa_generation.llm_qa_generator_batch collect <batch_id> output.jsonl mapping.csv qa_pairs.csv
```

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
Best for fast feedback with Groq/Nebius/Llama.
```bash
# Edit configs/serial/groq_run.yaml
# Defines model: "llama-3.3-70b-versatile", concurrency: 5
bash scripts/run_serial.sh configs/serial/groq_run.yaml
```

### 4. Analyze Results
Launch the analysis notebook to compare performance across experiments.
```bash
jupyter notebook notebooks/llm_benchmark_analysis.ipynb
```

**Quick Start Snippet:**
```python
from src.evaluation.analysis import analyze_results

# Aggregate metrics from Batch and Serial runs
results = analyze_results([
    "data/results/llm_batch", 
    "data/results/serial"
])
# Returns: DataFrame with Accuracy and Hallucination Rate for RAG, Knowledge, Adversarial layers
```



## Acknowledgments

- **Arelle Project**: Open-source XBRL parsing library
- **Financial Conduct Authority**: UK regulatory filing data
- **OpenAI/Groq/Nebius**: LLM API providers