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
