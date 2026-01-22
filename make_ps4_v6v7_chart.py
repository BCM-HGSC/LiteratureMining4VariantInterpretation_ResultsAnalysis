#!/usr/bin/env python3
"""
Make PS4 v6 vs v7 bar chart (exact-match rate),
analogous to your existing var_found v6/v7 comparison figure.

Reads:
    output/ps4_counts/ps4_v6_v7_core_metrics.xlsx
      - sheet: "core_ps4_v6v7"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Colors & aliases to roughly match your style
COLOR_V6 = "#5E3C99"   # purple-ish
COLOR_V7 = "#1B9E77"   # teal-ish
ECOLOR   = "#666666"

ALIAS = {
    "gemini": "Gemini 2.5 Pro",
    "gpt5":   "OpenAI GPT-5",
    "o3high": "OpenAI o3",
    "o4mini": "OpenAI o4-mini",
    "claude": "Claude Sonnet 4",
}


def plot_ps4_v6_v7_bar(
    xlsx_path: str,
    out_png: Path,
    fixed_order: str | None = None,
):
    core = pd.read_excel(xlsx_path, sheet_name="core_ps4_v6v7")

    # Expect columns: version, base_model, N, Exact_Match, Exact_Match_Rate, Acc_CI_low, Acc_CI_high
    needed = {"version", "base_model", "N", "Exact_Match", "Exact_Match_Rate", "Acc_CI_low", "Acc_CI_high"}
    missing = needed.difference(core.columns)
    if missing:
        raise ValueError(f"Missing columns in core_ps4_v6v7: {missing}")

    # Ensure correct dtype
    core["N"] = core["N"].astype(int)
    core["Exact_Match"] = core["Exact_Match"].astype(int)
    core["Exact_Match_Rate"] = core["Exact_Match_Rate"].astype(float)
    core["Acc_CI_low"] = core["Acc_CI_low"].astype(float)
    core["Acc_CI_high"] = core["Acc_CI_high"].astype(float)

    # Get v6 and v7 rows
    v6 = core[core["version"] == "v6"].copy()
    v7 = core[core["version"] == "v7"].copy()

    # We'll only plot models that exist in BOTH versions
    common_models = sorted(set(v6["base_model"]).intersection(v7["base_model"]))
    if fixed_order:
        order = [m.strip() for m in fixed_order.split(",") if m.strip() in common_models]
    else:
        # Default order: by v7 exact-match rate descending
        v7_sub = v7[v7["base_model"].isin(common_models)].copy()
        v7_sub = v7_sub.sort_values("Exact_Match_Rate", ascending=False)
        order = v7_sub["base_model"].tolist()

    v6 = v6[v6["base_model"].isin(order)].copy()
    v7 = v7[v7["base_model"].isin(order)].copy()

    # Index by base_model for easy alignment
    v6 = v6.set_index("base_model").loc[order]
    v7 = v7.set_index("base_model").loc[order]

    labels = [ALIAS.get(b, b) for b in order]
    x = np.arange(len(order))
    width = 0.36

    # Data
    N_v6 = v6["N"].to_numpy()
    N_v7 = v7["N"].to_numpy()

    exact_v6 = v6["Exact_Match"].to_numpy()
    exact_v7 = v7["Exact_Match"].to_numpy()

    rate_v6 = v6["Exact_Match_Rate"].to_numpy()
    rate_v7 = v7["Exact_Match_Rate"].to_numpy()

    ci_lo_v6 = v6["Acc_CI_low"].to_numpy()
    ci_hi_v6 = v6["Acc_CI_high"].to_numpy()
    ci_lo_v7 = v7["Acc_CI_low"].to_numpy()
    ci_hi_v7 = v7["Acc_CI_high"].to_numpy()

    # Convert to %
    y_v6 = rate_v6 * 100.0
    y_v7 = rate_v7 * 100.0

    yerr_v6 = np.vstack([(rate_v6 - ci_lo_v6) * 100.0, (ci_hi_v6 - rate_v6) * 100.0])
    yerr_v7 = np.vstack([(rate_v7 - ci_lo_v7) * 100.0, (ci_hi_v7 - rate_v7) * 100.0])

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(bottom=0.22)

    bars_v6 = ax.bar(
        x - width / 2,
        y_v6,
        width,
        label="v6",
        yerr=yerr_v6,
        capsize=5,
        ecolor=ECOLOR,
        color=COLOR_V6,
    )
    bars_v7 = ax.bar(
        x + width / 2,
        y_v7,
        width,
        label="v7",
        yerr=yerr_v7,
        capsize=5,
        ecolor=ECOLOR,
        color=COLOR_V7,
    )

    # Annotate bars with "a/b (c%)"
    for i in range(len(labels)):
        b6 = bars_v6[i]
        b7 = bars_v7[i]

        ax.text(
            b6.get_x() + b6.get_width() / 2,
            b6.get_height() / 2,
            f"{int(exact_v6[i])}/{int(N_v6[i])}\n({rate_v6[i]*100:.1f}%)",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
        ax.text(
            b7.get_x() + b7.get_width() / 2,
            b7.get_height() / 2,
            f"{int(exact_v7[i])}/{int(N_v7[i])}\n({rate_v7[i]*100:.1f}%)",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    ax.set_title("A. Task 2 (PS4 case count) model performance - v6 -> v7 comparison ",
                 fontsize=18, pad=15)
    ax.set_ylabel("Agreement with truth-set (%)", fontsize=12)
    ax.set_xticks(x, labels, rotation=30, ha="right")

    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=11)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PS4 v6/v7 bar chart → {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xlsx",
        required=True,
        help="Path to ps4_v6_v7_core_metrics.xlsx (core_ps4_v6v7 sheet)",
    )
    ap.add_argument(
        "--outpng",
        default="output/ps4_counts/ps4_v6_v7_bar.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--order",
        default="gemini,gpt5,o3high,o4mini,claude",
        help="Optional comma-separated base model order",
    )
    args = ap.parse_args()

    plot_ps4_v6_v7_bar(args.xlsx, Path(args.outpng), args.order)


if __name__ == "__main__":
    main()

"""
python make_ps4_v6v7_chart.py \
  --xlsx output/ps4_counts/ps4_v6_v7_core_metrics.xlsx \
  --outpng output/ps4_counts/ps4_v6_v7_ps4_exactmatch.png \
  --order gemini,gpt5,o3high,o4mini,claude

"""