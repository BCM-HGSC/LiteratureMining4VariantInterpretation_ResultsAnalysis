# Results Analysis: Literature Mining for ACMG/AMP/VCEP Guided PS4 Variant Interpretation Criteria using LLMs

This repository contains the input files and the code to generate output results and charts for 
PS4 evidence extraction from literature using Large Language Models (LLMs).
This readme documents the input files, code and resulting output. Results from five LLMs were analysed for 
two tasks - Task 1 - Variant Detection and Task 2 - PS4 Case Count

# LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7

Standalone analysis + plotting utilities for the v6/v7 PS4 and variant-found evaluations. Everything runs from plain Excel inputs; figures and tables are written under `LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output/` by default.

## 1) Input expectations
Combined truth/prediction workbook is in `LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/input/`. The analyzer expects these columns (names must match):

- Boolean truth: `truth_var_found` (or `var_found_truth`)
- Integer truth: `truth_ps4_count` (or `truth`)
- Per-model predictions: `v6_<model>_var_found`, `v7_<model>_var_found`, `v6_<model>_ps4_count`, `v7_<model>_ps4_count`
- Optional PS4 error text: `v7_<model>_ps4_count_error` (used to override exact matches if the model self-reports an error)
- `<model>` is one of: `gemini`, `gpt5`, `o3high`, `o4mini`, `claude`

You can keep your own file name; the examples below use `ps4_nocontrolsv6v7.xlsx`.

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
- If you want to start fresh, delete everything under `LiteratureMining4VariantInterpretation_ResultsAnalysis_V6V7/output/` and re-run step 3.
