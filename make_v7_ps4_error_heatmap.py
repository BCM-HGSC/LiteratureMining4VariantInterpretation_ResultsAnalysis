#!/usr/bin/env python3
"""
Generate the v7 PS4 model-error heatmap.

Input:
    output/ps4_counts/v7/ps4_v7_metrics.xlsx
        - sheet: per_publication_v7
        - columns: v7_<model>_ps4_count_error

Output:
    output/ps4_counts/v7/ps4_v7_error_heatmap.png
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Model order (confirmed by user)
# ---------------------------------------------------------
MODELS = [
    "v7_claude",
    "v7_o4mini",
    "v7_o3high",
    "v7_gpt5",
    "v7_gemini",
]

DISPLAY_NAMES = {
    "v7_claude": "Claude Sonnet 4",
    "v7_o4mini": "OpenAI o4-mini",
    "v7_o3high": "OpenAI o3",
    "v7_gpt5": "OpenAI GPT-5",
    "v7_gemini": "Gemini 2.5 Pro",
}

# ---------------------------------------------------------
# Raw → Category mapping (EXACT from your table)
# ---------------------------------------------------------
RAW_TO_CAT = {
    "Model error. Non-conservative counting for relatedness": "Overcount",
    "Model error. Variant not found": "Variant not found",
    "Model error. Phenotype mismatch": "Phenotype association",
    "Model error. Error in Recessive/Compound Het/Homozygous assessment": "Inheritance / Zygosity error",
    "Model error. Incorrect guideline interpretation": "Guideline interpretation error",
    "Model error. Overly conservative counting": "Undercount",
    "Model error. Not a new case report": "Duplicate case",
    "Model error. Incorrect inclusion from previously reported case": "Duplicate case",
    "Model error. Not a case report": "Not a case report",
    "Model error. From DB, not primary case": "Not a case report",
    "Model error. Is a case report": "Not recognized as case",
    "Model error. From DB but counted": "Not recognized as case",
    "Model error. Incorrect case identification": "Missed Evidence",
    "Model error. Incorrect counting": "Missed Evidence",
    "Model error. Unclear": "Unknown",
}

ALL_CATS = [
    "Overcount",
    "Variant not found",
    "Phenotype association",
    "Inheritance / Zygosity error",
    "Guideline interpretation error",
    "Undercount",
    "Duplicate case",
    "Missed Evidence",
    "Not a case report",
    "Unknown",        # ONLY for "Model error. Unclear"
    "Not recognized as case",
]

def map_error_to_category(raw):
    """Return category or None.
       - Empty/NaN/unknown → None
       - Only exact match 'Model error. Unclear' → 'Unknown'
       - All other known raw errors → mapped category
    """
    if raw is None:
        return None

    if not isinstance(raw, str):
        return None

    raw_clean = raw.strip()
    if raw_clean == "":
        return None

    # Exact match in mapping
    if raw_clean in RAW_TO_CAT:
        return RAW_TO_CAT[raw_clean]

    # Anything else → ignore
    return None

# ---------------------------------------------------------
# Main heatmap builder
# ---------------------------------------------------------
def build_error_heatmap(df: pd.DataFrame, outpath: Path):

    matrix = pd.DataFrame(0, index=ALL_CATS, columns=MODELS)

    for mk in MODELS:
        err_col = f"{mk}_ps4_count_error"
        if err_col not in df.columns:
            print(f"[WARN] Missing error column: {err_col} — skipping this model.")
            continue

        errs = df[err_col].astype("string", errors="ignore")

        # Map to categories
        cats = errs.map(map_error_to_category)

        # Remove ignored rows (None)
        cats = cats.dropna()

        # Count categories
        vc = cats.value_counts()

        for cat, ct in vc.items():
            if cat in matrix.index:
                matrix.loc[cat, mk] = ct

    # Prepare label with model totals
    col_labels = [
        f"{DISPLAY_NAMES[mk]} ({int(matrix[mk].sum())})" for mk in MODELS
    ]

    data = matrix.to_numpy()

    plt.figure(figsize=(12, max(5, len(ALL_CATS) * 0.45)))
    fig, ax = plt.subplots(figsize=(12, max(5, len(ALL_CATS) * 0.45)))

    im = ax.imshow(data, cmap="viridis", aspect="auto")

    ax.set_yticks(np.arange(len(ALL_CATS)))
    ax.set_yticklabels(ALL_CATS)

    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")

    # Write counts
    for i in range(len(ALL_CATS)):
        for j in range(len(MODELS)):
            v = int(data[i, j])
            if v > 0:
                rgba = im.cmap(im.norm(data[i, j]))
                r, g, b, _ = rgba
                luma = 0.299*r + 0.587*g + 0.114*b
                txt_color = "black" if luma > 0.6 else "white"
                ax.text(j, i, str(v), ha="center", va="center", color=txt_color, fontsize=10)

    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title("Comparative Model Error Distribution", fontsize=14, pad=12)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved heatmap → {outpath}")

# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--outpng",
        default="output/ps4_counts/v7/ps4_v7_error_heatmap.png")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="per_publication_v7")
    build_error_heatmap(df, Path(args.outpng))

if __name__ == "__main__":
    main()



"""
python make_v7_ps4_error_heatmap.py \
  --xlsx output/ps4_counts/v7/ps4_v7_metrics.xlsx \
  --outpng output/ps4_counts/v7/ps4_v7_error_heatmap.png

"""