#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------
# Paths
# ------------------------------
METRICS_PATH = "output/ps4_counts/v7/ps4_v7_metrics.xlsx"
PAIRWISE_PATH = "output/ps4_counts/v7_model_compare/ps4_v7_model_pairwise_exact.xlsx"
OUTDIR = Path("output/ps4_counts/v7/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Model mappings & order
# ------------------------------
ORDER = ["gemini", "gpt5", "o3high", "o4mini", "claude" ]

DISPLAY = {
    "gemini": "Gemini 2.5 Pro",
    "gpt5": "OpenAI GPT-5",
    "o3high": "OpenAI o3",
    "o4mini": "OpenAI o4-mini",
    "claude": "Claude Sonnet 4",
}

# ------------------------------
# Helper: significance stars
# ------------------------------
def p_to_star(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

# ------------------------------
# FIGURE 1 — Horizontal Bar Leaderboard
# ------------------------------
def make_leaderboard():
    df = pd.read_excel(METRICS_PATH, sheet_name="core_v7_metrics")
    df = df[df["base_model"].isin(ORDER)].copy()
    df["Display"] = df["base_model"].map(DISPLAY)

    # Sort by your custom order
    df["order"] = df["base_model"].apply(lambda m: ORDER.index(m))
    df = df.sort_values("order")

    # Extract values
    acc = df["Exact_Match_Rate"] * 100
    ci_low = df["Acc_CI_low"] * 100
    ci_high = df["Acc_CI_high"] * 100

    # Error bars = (upper - acc, acc - lower)
    err = np.vstack([acc - ci_low, ci_high - acc])

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(
        df["Display"],
        acc,
        xerr=err,
        capsize=6,
        color="#1f4e79",   # deep blue
        ecolor="gray",
        alpha=0.95
    )

    # Add text inside bars
    for i, row in df.iterrows():
        label = f"{row['Exact_Match']}/{row['N']} ({row['Exact_Match_Rate']*100:.1f}%)"
        ax.text(
            row["Exact_Match_Rate"] * 100 * 0.02 + 0.5,
            ORDER.index(row["base_model"]),
            label,
            va="center",
            ha="left",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_xlabel("PS4 case count - Concordance with truth-set (%)", fontsize=14)
    ax.set_title("A. PS4 case count – Model Leaderboard",
                 fontsize=18, pad=15)

    ax.set_xlim(0, 100)
    ax.invert_yaxis()  # highest performer at top
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    out = OUTDIR / "ps4_v7_leaderboard.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote leaderboard → {out}")


# ------------------------------
# FIGURE 2 — Pairwise Heatmap (unchanged)
# ------------------------------
def make_heatmap():
    df = pd.read_excel(PAIRWISE_PATH, sheet_name="v7_pairwise")

    gmat = pd.DataFrame(index=ORDER, columns=ORDER, dtype=float)
    pmat = pd.DataFrame(index=ORDER, columns=ORDER, dtype=float)

    for _, row in df.iterrows():
        A = row["Model_A"]
        B = row["Model_B"]
        g = row["Cohens_g"]
        p = row["p_raw"]

        gmat.loc[A, B] = g
        gmat.loc[B, A] = -g
        pmat.loc[A, B] = p
        pmat.loc[B, A] = p

    fig, ax = plt.subplots(figsize=(9, 8))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        gmat.astype(float),
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor="black",
        xticklabels=[DISPLAY[m] for m in ORDER],
        yticklabels=[DISPLAY[m] for m in ORDER],
        cbar_kws={"label": "Cohen's g"},
        ax=ax
    )

    for i, A in enumerate(ORDER):
        for j, B in enumerate(ORDER):
            if i == j:
                text = ""
            else:
                p = pmat.loc[A, B]
                g = gmat.loc[A, B]
                star = p_to_star(p)
                text = f"p={p:.3f}{star}\n g={g:+.2f}"
            ax.text(j + 0.5, i + 0.5, text,
                    ha="center", va="center", fontsize=10)

    plt.title("C. PS4 case count: Paired McNemar p-value + Cohen's g Effect Size",
              fontsize=16, pad=20)
    plt.xlabel("Model B", fontsize=13)
    plt.ylabel("Model A", fontsize=13)

    out = OUTDIR / "ps4_v7_pairwise_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote heatmap → {out}")


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    make_leaderboard()
    make_heatmap()

"""
python make_ps4_v7_figures.py
"""
