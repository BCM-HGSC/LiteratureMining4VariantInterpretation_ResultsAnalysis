# Stochasticity

This folder contains analysis code and inputs/outputs for PS4 stochasticity.

## Layout
- `highcomplexitypubs/`: stochasticity analysis for high-complexity publications.
- `lowcomplexitypubs/`: stochasticity analysis for low-complexity publications ("nostochasticity" inputs).

Each subfolder follows the same pattern:
- `input/`: model run CSVs used for analysis.
- `output/`: generated per-PMID and summary CSVs, plots, and Excel exports.
- `*_checker.py`: computes stochasticity metrics from inputs.
- `*_plots.py`: generates plots and Excel workbooks from the output CSVs.
- `*_emb_cache.jsonl`: embedding cache used by the semantic similarity step.

## How to run
From the repo root:

High complexity:
```
python stochasticity/highcomplexitypubs/stochasticitychecker.py
python stochasticity/highcomplexitypubs/ps4_stochasticity_plots.py
```

Low complexity (nostochasticity):
```
python stochasticity/lowcomplexitypubs/nostochasticitychecker.py
python stochasticity/lowcomplexitypubs/ps4_nostochasticity_plots.py
```

## Notes
- The checker scripts require `OPENAI_API_KEY` for embeddings.
- Outputs are written to each subfolder's `output/` directory.
- If you move the folder, the scripts resolve paths relative to their file location.
