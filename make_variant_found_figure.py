#!/usr/bin/env python3
"""
Make v7 Task 1 (variant-detection) figures.

Outputs (relative to repo root):

  output/bool_var_found/v7/plots/v7_variant_found_accuracy.png
  output/bool_var_found/v7/plots/v7_variant_found_pairwise_heatmap.png

Inputs:

  output/bool_var_found/v7/v7_variant_found_metrics_with_effects.xlsx
    - sheet: v7_boot_metrics
      cols used: base_model, N, TP, TN, Accuracy,
                  Acc_CI_low, Acc_CI_high

  output/bool_var_found/v7_model_compare/v7_boolean_model_pairwise.xlsx
    - sheet: v7_pairwise
      cols used: Model_A, Model_B,
                 A_correct_only, B_correct_only,
                 p_raw, Cohens_g
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

METRICS_PATH = Path("output/bool_var_found/v7/v7_variant_found_metrics_with_effects.xlsx")
PAIRWISE_PATH = Path("output/bool_var_found/v7_model_compare/v7_boolean_model_pairwise.xlsx")
PLOTS_DIR = Path("output/bool_var_found/v7/plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Model order + pretty names
# ---------------------------------------------------------------------

MODEL_ORDER = ["gemini", "gpt5", "o3high", "claude", "o4mini"]

DISPLAY_NAMES = {
    "gemini": "Gemini 2.5 Pro",
    "gpt5": "OpenAI GPT-5",
    "o3high": "OpenAI o3",       # NOTE: lower-case "o3"
    "claude": "Claude Sonnet 4",
    "o4mini": "OpenAI o4-mini",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_v7_metrics() -> pd.DataFrame:
    print(f"[INFO] Loading accuracy metrics from: {METRICS_PATH}")
    df = pd.read_excel(METRICS_PATH, sheet_name="v7_boot_metrics")
    # keep only rows we care about and enforce model order
    df = df[df["base_model"].isin(MODEL_ORDER)].copy()
    df["base_model"] = pd.Categorical(df["base_model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("base_model")
    return df


def load_v7_pairwise() -> pd.DataFrame:
    print(f"[INFO] Loading pairwise comparison from: {PAIRWISE_PATH}")
    df = pd.read_excel(PAIRWISE_PATH, sheet_name="v7_pairwise")
    return df


def stars_from_p(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------
# Figure A: Variant-detection accuracy bar plot (v7 only)
# ---------------------------------------------------------------------


def make_figure_A(df: pd.DataFrame):
    """
    Single-bar accuracy plot (matches truth only) with gray CIs
    and centered labels like:

        256/281
        (91.1%)
    """
    # Aggregate metrics
    base_models = df["base_model"].tolist()
    N = df["N"].to_numpy()
    TP = df["TP"].to_numpy()
    TN = df["TN"].to_numpy()
    acc = df["Accuracy"].to_numpy()
    lo = df["Acc_CI_low"].to_numpy()
    hi = df["Acc_CI_high"].to_numpy()

    matches = TP + TN
    acc_pct = acc * 100.0
    lo_pct = lo * 100.0
    hi_pct = hi * 100.0

    err_lower = acc_pct - lo_pct
    err_upper = hi_pct - acc_pct

    x = np.arange(len(base_models))

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # main bars (matches truth only)
    bar_color = "#1f4e79"  # dark blue
    bars = ax.bar(
        x,
        acc_pct,
        color=bar_color,
        width=0.6,
    )

    # gray CI error bars
    ax.errorbar(
        x,
        acc_pct,
        yerr=[err_lower, err_upper],
        fmt="none",
        ecolor="gray",
        elinewidth=1.5,
        capsize=4,
    )

    # centered labels inside bars
    for i, b in enumerate(bars):
        h = b.get_height()
        label = f"{int(matches[i])}/{int(N[i])}\n({acc_pct[i]:.1f}%)"
        ax.text(
            b.get_x() + b.get_width() / 2,
            h / 2,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    # x-axis labels
    xticklabels = [DISPLAY_NAMES[m] for m in base_models]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=25, ha="right", fontsize=10)

    ax.set_ylim(0, max(100, np.max(hi_pct) + 5))
    ax.set_ylabel("Variant-detection accuracy vs truth (%)", fontsize=11)

    ax.set_title(
        "Model Outcomes vs Ground Truth (Variant Detection, v7)\n"
        f"(N = {int(N[0])} publications)",
        fontsize=13,
        pad=12,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    outpath = PLOTS_DIR / "v7_variant_found_accuracy.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote Figure A → {outpath}")


# ---------------------------------------------------------------------
# Figure B: Pairwise v7 model comparison heatmap
# ---------------------------------------------------------------------


def make_figure_B(df_pw: pd.DataFrame):
    """
    Heatmap-style pairwise comparison of v7 models.

    - Rows: Model A
    - Columns: Model B
    - Color:
        * green if row model significantly better (p<0.05 & g<0)
        * red   if column model significantly better (p<0.05 & g>0)
        * white / light gray if not significant
    - Text:
        line 1: "p = 0.004**" or "ns" (if >=0.05)
        line 2: "g = 0.42" (Cohen's g with sign from row's perspective)
    """
    short_names = MODEL_ORDER
    n = len(short_names)

    # matrices for p-values and effect sizes in row/col orientation
    p_mat = np.full((n, n), np.nan)
    g_mat = np.full((n, n), np.nan)

    for i, mi in enumerate(short_names):
        for j, mj in enumerate(short_names):
            if mi == mj:
                continue

            # try A=mi, B=mj
            row = df_pw[(df_pw["Model_A"] == mi) & (df_pw["Model_B"] == mj)]
            flip = False
            if row.empty:
                # try reversed
                row = df_pw[(df_pw["Model_A"] == mj) & (df_pw["Model_B"] == mi)]
                flip = True

            if row.empty:
                continue

            row = row.iloc[0]
            p = float(row["p_raw"])
            g_raw = float(row["Cohens_g"])

            # orientation: p same; g sign depends on orientation
            # In the original table, g = (B_only - A_only)/(A_only+B_only).
            # We want g>0 = column (mj) better; g<0 = row (mi) better.
            if flip:
                g = -g_raw
            else:
                g = g_raw

            p_mat[i, j] = p
            g_mat[i, j] = g

    # plotting
    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    # base background
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()

    for i in range(n):
        for j in range(n):
            if i == j:
                # diagonal = light gray
                facecolor = "#f0f0f0"
                rect = Rectangle(
                    (j, i), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=1.0
                )
                ax.add_patch(rect)
                continue

            p = p_mat[i, j]
            g = g_mat[i, j]

            # default: neutral
            facecolor = "white"

            if not np.isnan(p) and p < 0.05:
                # significant — color by winner
                if g < 0:
                    # row model better → green
                    facecolor = "#c7e9c0"
                elif g > 0:
                    # column model better → red/pink
                    facecolor = "#fcbba1"
            else:
                # non-significant: very light gray
                facecolor = "#fdfdfd"

            rect = Rectangle(
                (j, i), 1, 1, facecolor=facecolor, edgecolor="#cccccc", linewidth=0.8
            )
            ax.add_patch(rect)

            # Text annotations
            if np.isnan(p) or np.isnan(g):
                continue

            if p >= 0.05:
                line1 = "ns"
            else:
                line1 = f"p={p:.3f}{stars_from_p(p)}"
            line2 = f"g={g:+.2f}"

            ax.text(
                j + 0.5,
                i + 0.40,
                line1,
                ha="center",
                va="center",
                fontsize=9,
            )
            ax.text(
                j + 0.5,
                i + 0.70,
                line2,
                ha="center",
                va="center",
                fontsize=9,
            )

    # ticks and labels
    xticklabels = [DISPLAY_NAMES[m] for m in short_names]
    yticklabels = [DISPLAY_NAMES[m] for m in short_names]

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(xticklabels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(yticklabels, fontsize=9)

    ax.set_xlabel("Model B", fontsize=11)
    ax.set_ylabel("Model A", fontsize=11)
    ax.set_title(
        "v7 Variant-detection: Pairwise McNemar (raw p-values)\n"
        "Green: row better, Red: column better",
        fontsize=12,
        pad=10,
    )

    fig.tight_layout()
    outpath = PLOTS_DIR / "v7_variant_found_pairwise_heatmap.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote Figure B → {outpath}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    df_metrics = load_v7_metrics()
    make_figure_A(df_metrics)

    df_pw = load_v7_pairwise()
    make_figure_B(df_pw)


if __name__ == "__main__":
    main()

"""
python make_variant_found_figure.py

"""