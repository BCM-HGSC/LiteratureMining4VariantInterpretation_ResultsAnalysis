#!/usr/bin/env python3
"""
All-in-one PS4 / var_found analysis for v6 & v7.

This script:

1. Computes Task 1 (var_found) metrics per version (v6, v7)
   - Accuracy, sensitivity, specificity, PPV, NPV, F1, MCC
   - 95% bootstrap CIs

2. Computes Task 2 (PS4 count) metrics per version (v6, v7)
   - Exact-match rate with Wilson 95% CI
   - Undercount / overcount frequencies

3. Performs PS4 v6→v7 comparison per model
   - Exact-match rates for v6 and v7
   - McNemar test (paired) with Holm correction
   - Cohen's g effect size

4. Performs PS4 v7-only model-vs-model comparison
   - Pairwise McNemar tests between v7 models
   - Holm-corrected p-values and Cohen's g

Outputs (relative to --outdir):

  bool_var_found/
    v6/v6_variant_found_metrics_with_effects.xlsx
    v7/v7_variant_found_metrics_with_effects.xlsx
    combined_boolean_core_metrics.xlsx
    v6_v7_compare/var_found_v6_v7_pairwise.xlsx

  ps4_counts/
    v6/ps4_v6_metrics.xlsx
    v7/ps4_v7_metrics.xlsx
    ps4_v6_v7_core_metrics.xlsx
    v6_v7_compare/ps4_v6_v7_exact_pairwise.xlsx
    v7_model_compare/ps4_v7_model_pairwise_exact.xlsx

Expected columns in --raw Excel:

  Truth:
    truth_var_found        # boolean truth for variant found
    truth_ps4_count        # integer truth for PS4 count

  Models (variant found):
    v6_o3high_var_found, v7_o3high_var_found
    v6_gemini_var_found,   v7_gemini_var_found
    v6_claude_var_found,   v7_claude_var_found
    v6_gpt5_var_found,     v7_gpt5_var_found
    v6_o4mini_var_found,   v7_o4mini_var_found

  Models (PS4 count):
    v6_o3high_ps4_count, v7_o3high_ps4_count
    v6_gemini_ps4_count, v7_gemini_ps4_count
    v6_claude_ps4_count, v7_claude_ps4_count
    v6_gpt5_ps4_count,   v7_gpt5_ps4_count
    v6_o4mini_ps4_count, v7_o4mini_ps4_count

  Optional PS4 error text columns (used when present):
    v7_<model>_ps4_count_error
    (v6 error columns are not expected but will be used if present)

Usage example:

  python ps4_all_in_one_v6v7.py \
      --raw ps4_nocontrolsv6v7.xlsx \
      --outdir output_v6v7
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODELS = ["gemini", "gpt5", "o3high", "o4mini", "claude"]
VERSIONS = ["v6", "v7"]

BOOL_TRUTH_CANDIDATES = ["truth_var_found", "var_found_truth"]
INT_TRUTH_CANDIDATES = ["truth_ps4_count", "truth"]

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ensure_parent(p: Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not pd.isna(x):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    return np.nan


def to_int(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score 95% CI for a proportion."""
    if n is None or n <= 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + (z ** 2) / n
    center = p + (z ** 2) / (2 * n)
    adj = z * ((p * (1 - p) + (z ** 2) / (4 * n)) / n) ** 0.5
    lo = (center - adj) / denom
    hi = (center + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


def confusion_counts(y_true, y_pred) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def f1_from_ppv_sens(ppv, sens):
    if any(pd.isna([ppv, sens])) or (ppv + sens) == 0:
        return np.nan
    return 2 * ppv * sens / (ppv + sens)


def mcc_from_counts(tp, fp, fn, tn):
    a = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if a == 0:
        return np.nan
    return ((tp * tn) - (fp * fn)) / (a ** 0.5)


def bootstrap_ci_pairs(y_true, y_pred, reps: int = 1000) -> Dict[str, float]:
    """Bootstrap CIs for accuracy, sens, spec, PPV, NPV, F1."""
    N = len(y_true)
    idx = np.arange(N)
    acc, sens, spec, ppv, npv, f1 = [], [], [], [], [], []
    for _ in range(reps):
        bs = RNG.choice(idx, size=N, replace=True)
        yt, yp = y_true[bs], y_pred[bs]
        tp, tn, fp, fn = confusion_counts(yt, yp)
        acc.append((tp + tn) / N)
        sens.append((tp / (tp + fn)) if (tp + fn) > 0 else np.nan)
        spec.append((tn / (tn + fp)) if (tn + fp) > 0 else np.nan)
        ppv.append((tp / (tp + fp)) if (tp + fp) > 0 else np.nan)
        npv.append((tn / (tn + fn)) if (tn + fn) > 0 else np.nan)
        ppv_b, sens_b = ppv[-1], sens[-1]
        f1.append(f1_from_ppv_sens(ppv_b, sens_b))

    pct = lambda arr: (np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5))
    return {
        "Acc_CI_low": pct(acc)[0],
        "Acc_CI_high": pct(acc)[1],
        "Sens_CI_low": pct(sens)[0],
        "Sens_CI_high": pct(sens)[1],
        "Spec_CI_low": pct(spec)[0],
        "Spec_CI_high": pct(spec)[1],
        "PPV_CI_low": pct(ppv)[0],
        "PPV_CI_high": pct(ppv)[1],
        "NPV_CI_low": pct(npv)[0],
        "NPV_CI_high": pct(npv)[1],
        "F1_CI_low": pct(f1)[0],
        "F1_CI_high": pct(f1)[1],
    }


def mcnemar_cc(a_only: int, b_only: int) -> Tuple[float, float]:
    """
    Continuity-corrected McNemar test (chi^2 approximation).
    a_only: # cases correct for A only
    b_only: # cases correct for B only
    Returns (statistic, p-value).
    """
    if (a_only + b_only) == 0:
        return 0.0, 1.0
    stat = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
    # 1-df chi-square tail ~ exp(-stat/2)
    p = float(np.exp(-stat / 2.0))
    return float(stat), p


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        mult = m - rank
        val = min(1.0, pvals[idx] * mult)
        running_max = max(running_max, val)
        adj[idx] = running_max
    out = np.empty(m, dtype=float)
    out[:] = adj[np.argsort(order)]
    return out


def cohens_g(a_only: int, b_only: int) -> float:
    """Cohen's g for paired binary outcomes."""
    denom = a_only + b_only
    if denom == 0:
        return 0.0
    return (b_only - a_only) / denom


# ---------------------------------------------------------------------
# Schema A: Boolean "variant found" analysis
# ---------------------------------------------------------------------


def analyze_boolean_for_version(
    df: pd.DataFrame,
    truth_col: str,
    version: str,
    outdir: Path,
) -> pd.DataFrame:
    """
    Run boolean (Task 1) analysis for all models in a given version (v6 or v7).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    model_cols = [f"{version}_{m}_var_found" for m in MODELS if f"{version}_{m}_var_found" in df.columns]
    if not model_cols:
        print(f"[Schema A] No boolean model columns found for {version}; skipping.")
        return pd.DataFrame()

    truth = df[truth_col].map(to_bool)

    metrics_rows = []
    conf_tabs = []

    for col in model_cols:
        base_model = col.replace(f"{version}_", "").replace("_var_found", "")
        pred = df[col].map(to_bool)
        keep = truth.notna() & pred.notna()
        yt = truth[keep].astype(int).to_numpy()
        yp = pred[keep].astype(int).to_numpy()
        N = len(yt)
        if N == 0:
            continue

        tp, tn, fp, fn = confusion_counts(yt, yp)
        acc = (tp + tn) / N
        sens = (tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        ppv = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        npv = (tn / (tn + fn)) if (tn + fn) > 0 else np.nan
        f1 = f1_from_ppv_sens(ppv, sens)
        mcc = mcc_from_counts(tp, fp, fn, tn)
        cis = bootstrap_ci_pairs(yt, yp, reps=1000)

        metrics_rows.append(
            {
                "version": version,
                "model_col": col,
                "base_model": base_model,
                "N": N,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Accuracy": acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "PPV": ppv,
                "NPV": npv,
                "F1": f1,
                "MCC": mcc,
                **cis,
            }
        )

        cm = pd.crosstab(pd.Series(yt, name="Truth"), pd.Series(yp, name="Prediction"))
        cm["model_col"] = col
        cm["version"] = version
        conf_tabs.append(cm.reset_index())

    metrics_df = pd.DataFrame(metrics_rows).sort_values("Accuracy", ascending=False)
    confusion_df = pd.concat(conf_tabs, ignore_index=True) if conf_tabs else pd.DataFrame()

    xlsx = ensure_parent(outdir / f"{version}_variant_found_metrics_with_effects.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        metrics_df.to_excel(xw, sheet_name=f"{version}_boot_metrics", index=False)
        confusion_df.to_excel(xw, sheet_name="confusion_matrices", index=False)

    print(f"[Schema A] Wrote boolean metrics for {version} → {xlsx}")
    return metrics_df


def compare_boolean_v6_v7(
    df: pd.DataFrame,
    truth_col: str,
    outdir: Path,
) -> pd.DataFrame:
    """
    For each base model, compare v6 vs v7 boolean accuracy using McNemar + Cohen's g.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    truth = df[truth_col].map(to_bool)

    rows = []
    for m in MODELS:
        col_v6 = f"v6_{m}_var_found"
        col_v7 = f"v7_{m}_var_found"
        if col_v6 not in df.columns or col_v7 not in df.columns:
            continue

        a_pred = df[col_v6].map(to_bool)
        b_pred = df[col_v7].map(to_bool)

        keep = truth.notna() & a_pred.notna() & b_pred.notna()
        yt = truth[keep].astype(int).to_numpy()
        a_ok = (a_pred[keep].astype(int).to_numpy() == yt)
        b_ok = (b_pred[keep].astype(int).to_numpy() == yt)

        N = int(keep.sum())
        a_acc = float(a_ok.mean()) if N else np.nan
        b_acc = float(b_ok.mean()) if N else np.nan

        a_only = int(((a_ok == 1) & (b_ok == 0)).sum())
        b_only = int(((a_ok == 0) & (b_ok == 1)).sum())
        both_correct = int(((a_ok == 1) & (b_ok == 1)).sum())
        both_wrong = int(((a_ok == 0) & (b_ok == 0)).sum())

        stat, p = mcnemar_cc(a_only, b_only)
        g = cohens_g(a_only, b_only)

        rows.append(
            {
                "base_model": m,
                "N": N,
                "v6_column": col_v6,
                "v7_column": col_v7,
                "v6_exact_rate": a_acc,
                "v7_exact_rate": b_acc,
                "v7_minus_v6_exact_rate": (b_acc - a_acc) if N else np.nan,
                "v6_correct_only": a_only,
                "v7_correct_only": b_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "McNemar_stat_cc": stat,
                "p_raw": p,
                "Cohens_g": g,
                "Cohens_g_abs": abs(g),
            }
        )

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df["p_holm"] = holm_correction(pair_df["p_raw"].to_numpy())
    else:
        pair_df["p_holm"] = []

    xlsx = ensure_parent(outdir / "var_found_v6_v7_pairwise.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        pair_df.to_excel(xw, sheet_name="v6_v7_pairwise", index=False)

    print(f"[Schema A] Wrote v6 vs v7 boolean comparison → {xlsx}")
    return pair_df

def compare_boolean_v7_models(
    df: pd.DataFrame,
    truth_col: str,
    outdir: Path,
) -> pd.DataFrame:
    """
    Pairwise comparison of v7 variant-found correctness using McNemar tests.
    Produces a v7-only leaderboard for Task 1.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    truth = df[truth_col].map(to_bool)

    rows = []

    # Loop through all model pairs
    for i in range(len(MODELS)):
        m_a = MODELS[i]
        col_a = f"v7_{m_a}_var_found"
        if col_a not in df.columns:
            continue

        pred_a = df[col_a].map(to_bool)

        for j in range(i + 1, len(MODELS)):
            m_b = MODELS[j]
            col_b = f"v7_{m_b}_var_found"
            if col_b not in df.columns:
                continue

            pred_b = df[col_b].map(to_bool)

            # Row-level alignment
            keep = truth.notna() & pred_a.notna() & pred_b.notna()
            if not keep.any():
                continue

            yt = truth[keep].astype(int).to_numpy()
            ya = pred_a[keep].astype(int).to_numpy()
            yb = pred_b[keep].astype(int).to_numpy()

            # Correctness arrays
            a_ok = (ya == yt)
            b_ok = (yb == yt)

            N = int(keep.sum())
            a_only = int(((a_ok == 1) & (b_ok == 0)).sum())
            b_only = int(((a_ok == 0) & (b_ok == 1)).sum())
            both_correct = int(((a_ok == 1) & (b_ok == 1)).sum())
            both_wrong = int(((a_ok == 0) & (b_ok == 0)).sum())

            # McNemar + Cohen's g
            stat, p_raw = mcnemar_cc(a_only, b_only)
            g = cohens_g(a_only, b_only)

            rows.append(
                {
                    "Model_A": m_a,
                    "Model_B": m_b,
                    "N": N,
                    "A_correct_only": a_only,
                    "B_correct_only": b_only,
                    "both_correct": both_correct,
                    "both_wrong": both_wrong,
                    "McNemar_stat_cc": stat,
                    "p_raw": p_raw,
                    "Cohens_g": g,
                    "Cohens_g_abs": abs(g),
                }
            )

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["p_holm"] = holm_correction(df_out["p_raw"].to_numpy())

    # Write results
    xlsx = ensure_parent(outdir / "v7_boolean_model_pairwise.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        df_out.to_excel(xw, sheet_name="v7_pairwise", index=False)

    print(f"[Schema A] Wrote v7 variant-found model-vs-model comparison → {xlsx}")
    return df_out


# ---------------------------------------------------------------------
# Schema B: PS4 count analysis
# ---------------------------------------------------------------------


def _find_error_column(df: pd.DataFrame, version: str, model: str) -> Optional[str]:
    """
    Helper: return '<version>_<model>_ps4_count_error' if present.
    """
    col = f"{version}_{model}_ps4_count_error"
    return col if col in df.columns else None


def analyze_ps4_for_version(
    df: pd.DataFrame,
    truth_col: str,
    version: str,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    PS4 count analysis for a given version (v6 or v7).

    Metrics per model:
      - Exact_Match (count)
      - Exact_Match_Rate (proportion) + Wilson CI
      - Mismatch
      - Undercount_mismatches
      - Overcount_mismatches

    Per-publication sheet includes:
      - truth_col
      - predicted counts
      - model-error flags (if any)
      - final_exact flags
    """
    outdir.mkdir(parents=True, exist_ok=True)
    truth = to_int(df[truth_col])

    metrics_rows = []
    per_pub_cols = [truth_col]

    for m in MODELS:
        val_col = f"{version}_{m}_ps4_count"
        if val_col not in df.columns:
            continue

        err_col = _find_error_column(df, version, m)
        pred = to_int(df[val_col])

        if err_col is not None:
            errtxt = df[err_col].astype("string").fillna("")
            errflag = errtxt.str.strip().str.lower().str.startswith("model error")
        else:
            errflag = pd.Series(False, index=df.index)

        keep = truth.notna() & pred.notna()
        yt = truth[keep].astype(int)
        yp = pred[keep].astype(int)

        numeric_equal = (yt == yp)
        final_exact = numeric_equal & (~errflag[keep])
        N = int(keep.sum())
        exact = int(final_exact.sum())
        mism = N - exact

        # Under/over counts among mismatches only
        diff = yp - yt
        under = int(((diff < 0) & (~final_exact)).sum())
        over = int(((diff > 0) & (~final_exact)).sum())

        if N:
            ci_lo, ci_hi = wilson_ci(exact, N)
            acc = exact / N
        else:
            ci_lo, ci_hi, acc = (np.nan, np.nan, np.nan)

        metrics_rows.append(
            {
                "version": version,
                "base_model": m,
                "model_col": val_col,
                "error_col": err_col if err_col else "",
                "N": N,
                "Exact_Match": exact,
                "Mismatch": mism,
                "Exact_Match_Rate": acc,
                "Acc_CI_low": ci_lo,
                "Acc_CI_high": ci_hi,
                "Undercount_mismatches": under,
                "Overcount_mismatches": over,
            }
        )

        # Per-publication diagnostics
        final_exact_col = pd.Series(pd.NA, index=df.index, dtype="object")
        final_exact_col.loc[keep] = final_exact.astype(int).values
        df[f"{val_col}_final_exact"] = final_exact_col

        per_pub_cols.append(val_col)
        if err_col is not None:
            per_pub_cols.append(err_col)
        per_pub_cols.append(f"{val_col}_final_exact")

    core = pd.DataFrame(metrics_rows).sort_values("Exact_Match_Rate", ascending=False)
    per_pub = df[per_pub_cols].copy()

    xlsx = ensure_parent(outdir / f"ps4_{version}_metrics.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        core.to_excel(xw, sheet_name=f"core_{version}_metrics", index=False)
        per_pub.to_excel(xw, sheet_name=f"per_publication_{version}", index=False)

    print(f"[Schema B] Wrote PS4 metrics for {version} → {xlsx}")
    return core, per_pub


def compare_ps4_v6_v7_exact(
    df: pd.DataFrame,
    truth_col: str,
    outdir: Path,
) -> pd.DataFrame:
    """
    Per-model PS4 comparison between v6 and v7 using ONLY exact-match
    (numeric equality + no model-error override).

    For each base model we compute:
      - exact-match counts and rates (v6 and v7)
      - McNemar test for paired correctness
      - Cohen's g effect size
    """
    outdir.mkdir(parents=True, exist_ok=True)
    truth = to_int(df[truth_col])

    rows = []
    for m in MODELS:
        col_v6 = f"v6_{m}_ps4_count"
        col_v7 = f"v7_{m}_ps4_count"
        if col_v6 not in df.columns or col_v7 not in df.columns:
            continue

        pred_v6 = to_int(df[col_v6])
        pred_v7 = to_int(df[col_v7])

        err_v6_col = _find_error_column(df, "v6", m)
        err_v7_col = _find_error_column(df, "v7", m)

        if err_v6_col is not None:
            err_v6_flag = df[err_v6_col].astype("string").fillna("").str.strip().str.lower().str.startswith("model error")
        else:
            err_v6_flag = pd.Series(False, index=df.index)

        if err_v7_col is not None:
            err_v7_flag = df[err_v7_col].astype("string").fillna("").str.strip().str.lower().str.startswith("model error")
        else:
            err_v7_flag = pd.Series(False, index=df.index)

        keep = truth.notna() & pred_v6.notna() & pred_v7.notna()
        if not keep.any():
            continue

        yt = truth[keep].astype(int)
        y6 = pred_v6[keep].astype(int)
        y7 = pred_v7[keep].astype(int)

        eq6 = (y6 == yt) & (~err_v6_flag[keep])
        eq7 = (y7 == yt) & (~err_v7_flag[keep])

        N = int(keep.sum())
        exact6 = int(eq6.sum())
        exact7 = int(eq7.sum())
        rate6 = exact6 / N
        rate7 = exact7 / N

        v6_only = int(((eq6 == 1) & (eq7 == 0)).sum())
        v7_only = int(((eq6 == 0) & (eq7 == 1)).sum())
        both_correct = int(((eq6 == 1) & (eq7 == 1)).sum())
        both_wrong = int(((eq6 == 0) & (eq7 == 0)).sum())

        stat, p = mcnemar_cc(v6_only, v7_only)
        g = cohens_g(v6_only, v7_only)

        rows.append(
            {
                "base_model": m,
                "N": N,
                "v6_column": col_v6,
                "v7_column": col_v7,
                "v6_exact": exact6,
                "v7_exact": exact7,
                "v6_exact_rate": rate6,
                "v7_exact_rate": rate7,
                "v7_minus_v6_exact_rate": rate7 - rate6,
                "v6_correct_only": v6_only,
                "v7_correct_only": v7_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "McNemar_stat_cc": stat,
                "p_raw": p,
                "Cohens_g": g,
                "Cohens_g_abs": abs(g),
            }
        )

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df["p_holm"] = holm_correction(pair_df["p_raw"].to_numpy())
    else:
        pair_df["p_holm"] = []

    xlsx = ensure_parent(outdir / "ps4_v6_v7_exact_pairwise.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        pair_df.to_excel(xw, sheet_name="v6_v7_exact_match", index=False)

    print(f"[Schema B] Wrote v6 vs v7 PS4 exact-match comparison → {xlsx}")
    return pair_df


def compare_ps4_v7_models(
    df: pd.DataFrame,
    truth_col: str,
    outdir: Path,
) -> pd.DataFrame:
    """
    Pairwise PS4 comparison between v7 models only, using exact-match
    correctness and McNemar tests.

    Produces a table suitable for a "leaderboard" style comparison.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    truth = to_int(df[truth_col])

    rows = []

    for i in range(len(MODELS)):
        m_a = MODELS[i]
        col_a = f"v7_{m_a}_ps4_count"
        if col_a not in df.columns:
            continue

        pred_a = to_int(df[col_a])
        err_a_col = _find_error_column(df, "v7", m_a)
        if err_a_col is not None:
            err_a_flag = df[err_a_col].astype("string").fillna("").str.strip().str.lower().str.startswith("model error")
        else:
            err_a_flag = pd.Series(False, index=df.index)

        for j in range(i + 1, len(MODELS)):
            m_b = MODELS[j]
            col_b = f"v7_{m_b}_ps4_count"
            if col_b not in df.columns:
                continue

            pred_b = to_int(df[col_b])
            err_b_col = _find_error_column(df, "v7", m_b)
            if err_b_col is not None:
                err_b_flag = df[err_b_col].astype("string").fillna("").str.strip().str.lower().str.startswith("model error")
            else:
                err_b_flag = pd.Series(False, index=df.index)

            keep = truth.notna() & pred_a.notna() & pred_b.notna()
            if not keep.any():
                continue

            yt = truth[keep].astype(int)
            ya = pred_a[keep].astype(int)
            yb = pred_b[keep].astype(int)

            ea = (ya == yt) & (~err_a_flag[keep])
            eb = (yb == yt) & (~err_b_flag[keep])

            N = int(keep.sum())
            a_only = int(((ea == 1) & (eb == 0)).sum())
            b_only = int(((ea == 0) & (eb == 1)).sum())
            both_correct = int(((ea == 1) & (eb == 1)).sum())
            both_wrong = int(((ea == 0) & (eb == 0)).sum())

            stat, p = mcnemar_cc(a_only, b_only)
            g = cohens_g(a_only, b_only)

            rows.append(
                {
                    "Model_A": m_a,
                    "Model_B": m_b,
                    "N": N,
                    "A_correct_only": a_only,
                    "B_correct_only": b_only,
                    "both_correct": both_correct,
                    "both_wrong": both_wrong,
                    "McNemar_stat_cc": stat,
                    "p_raw": p,
                    "Cohens_g": g,
                    "Cohens_g_abs": abs(g),
                }
            )

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df["p_holm"] = holm_correction(pair_df["p_raw"].to_numpy())
    else:
        pair_df["p_holm"] = []

    xlsx = ensure_parent(outdir / "ps4_v7_model_pairwise_exact.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        pair_df.to_excel(xw, sheet_name="v7_pairwise", index=False)

    print(f"[Schema B] Wrote v7-only PS4 model comparison → {xlsx}")
    return pair_df


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------


def detect_truth_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Input Excel (combined v6+v7)")
    ap.add_argument("--outdir", default="output_v6v7", help="Output directory root")
    ap.add_argument("--sheet", default=None, help="Optional sheet name")
    args = ap.parse_args()

    raw = Path(args.raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.sheet:
        df = pd.read_excel(raw, sheet_name=args.sheet)
    else:
        df = pd.read_excel(raw)

    # ---------------- Boolean / var_found (Schema A) ----------------
    bool_truth = detect_truth_column(df, BOOL_TRUTH_CANDIDATES)
    if bool_truth:
        print(f"[INFO] Using boolean truth column: {bool_truth}")
        bool_dir = outdir / "bool_var_found"

        core_bool_versions = []
        for v in VERSIONS:
            core_v = analyze_boolean_for_version(df.copy(), bool_truth, v, bool_dir / v)
            if not core_v.empty:
                core_bool_versions.append(core_v)

        if core_bool_versions:
            combined_bool_core = pd.concat(core_bool_versions, ignore_index=True)
            xlsx = ensure_parent(bool_dir / "combined_boolean_core_metrics.xlsx")
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
                combined_bool_core.to_excel(xw, sheet_name="combined_core", index=False)
            print(f"[Schema A] Wrote combined boolean core metrics → {xlsx}")

        compare_boolean_v6_v7(df.copy(), bool_truth, bool_dir / "v6_v7_compare")
        compare_boolean_v7_models(df.copy(), bool_truth, bool_dir / "v7_model_compare")
    else:
        print("[INFO] No boolean truth column found; skipping Schema A.")

    # ---------------- PS4 / integer (Schema B) ----------------
    int_truth = detect_truth_column(df, INT_TRUTH_CANDIDATES)
    if int_truth:
        print(f"[INFO] Using PS4 truth column: {int_truth}")
        ps4_dir = outdir / "ps4_counts"

        core_int_versions = []
        perpub_int_versions = {}
        for v in VERSIONS:
            core_v, per_pub_v = analyze_ps4_for_version(df.copy(), int_truth, v, ps4_dir / v)
            if not core_v.empty:
                core_int_versions.append(core_v)
                perpub_int_versions[v] = per_pub_v

        if core_int_versions:
            combined_int_core = pd.concat(core_int_versions, ignore_index=True)
            xlsx = ensure_parent(ps4_dir / "ps4_v6_v7_core_metrics.xlsx")
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
                combined_int_core.to_excel(xw, sheet_name="core_ps4_v6v7", index=False)
                for v, per_pub in perpub_int_versions.items():
                    per_pub.to_excel(xw, sheet_name=f"per_publication_{v}", index=False)
            print(f"[Schema B] Wrote combined PS4 core metrics → {xlsx}")

        compare_ps4_v6_v7_exact(df.copy(), int_truth, ps4_dir / "v6_v7_compare")
        compare_ps4_v7_models(df.copy(), int_truth, ps4_dir / "v7_model_compare")
    else:
        print("[INFO] No PS4 integer truth column found; skipping Schema B.")


if __name__ == "__main__":
    main()

"""
Example usage:

python ps4_all_in_one_v6v7.py \
  --raw input/ps4_nocontrolsv6v7.xlsx \
  --outdir output_v6v7
"""

