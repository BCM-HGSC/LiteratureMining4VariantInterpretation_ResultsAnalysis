#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Produce two figures (each saved twice):
  - Core stability metrics: grouped bars for variant detection, PS4 case count, reasoning stability
  - Overall stability index

Each figure is exported as:
  - a CLEAN version (no error bars) for slides
  - a SPREAD version (± std across PMIDs) for papers/tech talks

Outputs are saved as PNGs plus an Excel workbook (with embedded charts and the model summary table).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import xlsxwriter  # type: ignore # noqa: F401
except ImportError:  # pragma: no cover
    xlsxwriter = None

def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    ap = argparse.ArgumentParser()
    ap.add_argument("--per", default=str(output_dir / "stochasticity_only_per_pmid.csv"), help="per-PMID CSV")
    ap.add_argument("--summary", default=str(output_dir / "stochasticity_only_summary_by_model.csv"), help="by-model summary CSV")
    ap.add_argument("--prefix", default=str(output_dir / "ps4_stochasticity_"), help="output filename prefix")
    args = ap.parse_args()

    output_dir.mkdir(parents=True, exist_ok=True)

    per = pd.read_csv(args.per)
    summ = pd.read_csv(args.summary)

    # Recast run instability as stability (1 - unstable share)
    if "Pct_unstable_count" in per.columns:
        per["Pct_stable_count"] = 1 - per["Pct_unstable_count"]
    if "Pct_unstable_count" in summ.columns:
        summ["Pct_stable_count"] = 1 - summ["Pct_unstable_count"]

    # Ensure Variant_found_consistent is numeric (1/0) for std calculation
    per["Variant_found_consistent"] = (
        per["Variant_found_consistent"]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "t", "yes", "y"])
        .astype(float)
    )

    core_palette = plt.get_cmap("BuGn_r")(np.linspace(0.25, 0.85, 3))
    stability_color = plt.get_cmap("Blues")(0.55)

    metric_defs = {
        "pct_stable_count": {
            "per_col": "Pct_stable_count",
            "summary_col": "Pct_stable_count",
            "label": "Task 2: PS4 Case Count",
            "title": "PS4 Case Count Stability (Higher is Better)",
            "basename": "pct_stable_count",
            "color": core_palette[1],
        },
        "variant_found_consistent": {
            "per_col": "Variant_found_consistent",
            "summary_col": "pct_variant_found_consistent",
            "label": "Task 1:Variant Detection",
            "title": "Variant Detection Stability (Higher is Better)",
            "basename": "variant_found_consistent",
            "color": core_palette[2],
        },
        "semantic_mean_consolidated": {
            "per_col": "Semantic_mean_consolidated",
            "summary_col": "Semantic_mean_consolidated",
            "label": "Model Reasoning",
            "title": "Reasoning Stability (Higher is Better)",
            "basename": "semantic_mean_consolidated",
            "color": core_palette[0],
        },
        "stability_index": {
            "per_col": "Stability_index",
            "summary_col": "Stability_index",
            "label": "Overall stability index",
            "title": "Overall Stability Index",
            "basename": "stability_index",
            "color": "#006d77",
        },
    }

    charts = [
        {
            "metrics": [
                "variant_found_consistent",
                "pct_stable_count",
                "semantic_mean_consolidated",
            ],
            "title": "Core Stability Metrics",
            "spread_title": "Core Stability Metrics",
            "ylabel": "Reproducibility (0–1)",
            "basename": "core_metrics",
        },
        {
            "metrics": ["stability_index"],
            "title": metric_defs["stability_index"]["title"],
            "spread_title": metric_defs["stability_index"]["title"],
            "ylabel": "Reproducibility (0–1)",
            "basename": metric_defs["stability_index"]["basename"],
        },
    ]

    # Compute std across PMIDs per model from the per-PMID file
    stds_by_metric = {}
    for key, meta in metric_defs.items():
        per_col = meta["per_col"]
        if per_col in per.columns:
            stds_by_metric[key] = (
                per.groupby("Model")[per_col]
                   .std()
                   .reindex(summ["Model"])
                   .values
            )
        else:
            stds_by_metric[key] = None

    def annotate_bars(ax, bar_container):
        for bar in bar_container.patches:
            height = bar.get_height()
            if not np.isfinite(height):
                continue
            text = f"{height:.2f}"
            y = bar.get_y() + height / 2
            r, g, b, _ = bar.get_facecolor()
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "black" if luminance > 0.6 else "white"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold",
            )

    def resolve_color(spec):
        if isinstance(spec, np.ndarray):
            return spec
        if isinstance(spec, str):
            try:
                return plt.get_cmap(spec)(0.6)
            except ValueError:
                return spec
        return spec

    def upper_bound_for(values):
        return max(1.05, np.nanmax(values) * 1.05)

    highlight_color = "#1E5631"
    default_tick_color = "#102B54"
    highlight_labels = {
        "gpt-5",
        "openai gpt-5",
        "gemini 2.5 pro",
        "openai o3",
        "o3",
    }

    def colorize_xticks(ax):
        for tick in ax.get_xticklabels():
            text = tick.get_text()
            key = text.strip().lower()
            if key in highlight_labels:
                tick.set_color(highlight_color)
            else:
                tick.set_color(default_tick_color)

    def apply_legend(ax, fig, metric_count):
        if metric_count > 1:
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.0, -0.18),
                ncol=1,
                borderaxespad=0.0,
                frameon=True,
                framealpha=0.9,
                facecolor="white",
                labelspacing=0.2,
                handletextpad=0.4,
            )
            fig.subplots_adjust(bottom=0.28)

    # Plot each chart twice: clean + with std error bars
    saved = []
    chart_exports = []
    models = summ["Model"].values
    x_pos = np.arange(len(models))

    for chart in charts:
        metric_keys = chart["metrics"]
        n_metrics = len(metric_keys)
        bar_width = 0.8 / max(1, n_metrics)
        if n_metrics > 1:
            offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_metrics)
        else:
            offsets = [0.0]

        means_by_metric = {}
        errs_by_metric = {}
        colors_by_metric = {}
        for key in metric_keys:
            meta = metric_defs[key]
            means_by_metric[key] = summ[meta["summary_col"]].values
            errs_by_metric[key] = stds_by_metric.get(key)
            colors_by_metric[key] = meta.get("color", meta.get("cmap"))

        error_kwargs = {"ecolor": "#a6a6a6", "elinewidth": 1.2, "capsize": 5, "alpha": 0.9}

        # 1) CLEAN
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        tracked_means = []
        for offset, key in zip(offsets, metric_keys):
            means = means_by_metric[key]
            tracked_means.append(means)
            color = resolve_color(colors_by_metric[key])
            bars = ax.bar(
                x_pos + offset,
                means,
                bar_width,
                color=color,
                label=metric_defs[key]["label"],
            )
            annotate_bars(ax, bars)

        if tracked_means:
            peak_values = np.nanmax(np.vstack(tracked_means), axis=0)
            ax.set_ylim(0, upper_bound_for(peak_values))

        ax.set_ylabel(chart["ylabel"])
        #ax.set_title(chart["title"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=20)
        colorize_xticks(ax)
        apply_legend(ax, fig, n_metrics)
        fig.tight_layout()
        out_clean = f"{args.prefix}fig_{chart['basename']}__clean.png"
        fig.savefig(out_clean, dpi=200)
        plt.close(fig)
        saved.append(out_clean)

        # 2) WITH STD ERROR BARS
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        tracked_upper = []
        for offset, key in zip(offsets, metric_keys):
            means = means_by_metric[key]
            errs = errs_by_metric[key]
            color = resolve_color(colors_by_metric[key])
            if errs is not None:
                errs_clean = np.where(np.isnan(errs), 0, errs)
                bars = ax.bar(
                    x_pos + offset,
                    means,
                    bar_width,
                    yerr=errs_clean,
                    error_kw=error_kwargs,
                    color=color,
                    label=metric_defs[key]["label"],
                )
                tracked_upper.append(means + errs_clean)
            else:
                bars = ax.bar(
                    x_pos + offset,
                    means,
                    bar_width,
                    color=color,
                    label=metric_defs[key]["label"],
                )
                tracked_upper.append(means)
            annotate_bars(ax, bars)

        if tracked_upper:
            peak_values = np.nanmax(np.vstack(tracked_upper), axis=0)
            ax.set_ylim(0, upper_bound_for(peak_values))

        ax.set_ylabel(chart["ylabel"] + " (± std across PMIDs)")
        ax.set_title(chart["spread_title"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=20)
        colorize_xticks(ax)
        apply_legend(ax, fig, n_metrics)
        fig.tight_layout()
        out_err = f"{args.prefix}fig_{chart['basename']}__with_std.png"
        fig.savefig(out_err, dpi=200)
        plt.close(fig)
        saved.append(out_err)
        chart_exports.append({
            "basename": chart["basename"],
            "title": chart["title"],
            "spread_title": chart["spread_title"],
            "clean": out_clean,
            "spread": out_err,
        })

    excel_path = f"{args.prefix}stability_plots.xlsx"
    if xlsxwriter is None:
        print("xlsxwriter not installed; skipping Excel export.")
    else:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            summ.to_excel(writer, sheet_name="summary_by_model", index=False)
            workbook = writer.book
            for export in chart_exports:
                sheet_name = export["basename"][:31]
                worksheet = workbook.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet
                worksheet.set_column("A:A", 42)
                worksheet.write(0, 0, export["title"])
                worksheet.insert_image(1, 0, export["clean"], {
                    "x_scale": 0.75,
                    "y_scale": 0.75,
                })
                worksheet.write(28, 0, export["spread_title"])
                worksheet.insert_image(29, 0, export["spread"], {
                    "x_scale": 0.75,
                    "y_scale": 0.75,
                })
        saved.append(excel_path)

    print("Wrote:")
    for p in saved:
        print(" -", p)

if __name__ == "__main__":
    main()
