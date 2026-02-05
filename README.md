# Results Analysis: Literature Mining for ACMG/AMP/VCEP Guided PS4 Variant Interpretation Criteria using LLMs

This repository contains the input and output files and the code to generate output results and charts for 
PS4 evidence extraction from literature using Large Language Models (LLMs).
This readme documents the input files, code and resulting output. Results from five LLMs were analysed for 
two tasks - Task 1 - Variant Detection and Task 2 - PS4 Case Count

# LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7

Standalone analysis + plotting utilities for the v6/v7 PS4 and variant-found evaluations. Everything runs from plain Excel inputs; figures and tables are written under `LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output/` by default.

## 1) Input Files
Combined truth/prediction workbook is in `LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/input/`. 
1. ModelResultv6v7.xlsx contains the results from each of the five models, gpt-5, o3, o4-mini, gemini 2.5 pro and claude sonnet 4, for v6 and v7 prompt and output schema versions. ps4_nocontrolsv6v7.xlsx contains concordance with the truthset for the 281 publication-variant pairs for both tasks.
2. NegativeControlResultsAnalysis.xlsx contains the v7 results for each of the five models for the 28 negative controls.

The results analyzer code uses these columns from ps4_nocontrolsv6v7.xlsx for aggregate results analysis:
- Boolean truth: `truth_var_found` (or `var_found_truth`)
- Integer truth: `truth_ps4_count` (or `truth`)
- Per-model predictions: `v6_<model>_var_found`, `v7_<model>_var_found`, `v6_<model>_ps4_count`, `v7_<model>_ps4_count`
- Optional PS4 error text: `v7_<model>_ps4_count_error` (used to override exact matches if the model self-reports an error)
- `<model>` is one of: `gemini`, `gpt5`, `o3high`, `o4mini`, `claude`

## 2) Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/requirements.txt
```

## 3) Run the analysis (tables)
```bash
python LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/ps4_all_in_one_v6v7.py \
  --raw LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/input/ps4_nocontrolsv6v7.xlsx \
  --outdir LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output
```
- Use `--sheet SHEETNAME` if your workbook has multiple sheets.
- Outputs land under `output/bool_var_found/` and `output/ps4_counts/` with per-version metrics, v6→v7 comparisons, and v7-only pairwise comparisons.

## 4) Make publication figures (optional)
Run after the tables above are generated:
```bash
# PS4 v6 vs v7 bar chart
python LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/make_ps4_v6v7_chart.py \
  --xlsx LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output/ps4_counts/ps4_v6_v7_core_metrics.xlsx \
  --outpng LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output/ps4_counts/ps4_v6_v7_ps4_exactmatch.png

# PS4 v7 leaderboard + pairwise heatmap
python LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/make_ps4_v7_figures.py

# PS4 v7 error-category heatmap
python LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/make_v7_ps4_error_heatmap.py \
  --xlsx LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output/ps4_counts/v7/ps4_v7_metrics.xlsx

# Variant-found (Task 1) v7 figures
python LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/make_variant_found_figure.py
```

## 5) Notes
- Generated Excel/PNG artifacts are ignored by Git via the local `.gitignore`.
- To start fresh, delete everything under `LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output/` and re-run step 3.
