#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PS4 Stochasticity Analysis (Pct_unstable_count-based, with variant_found stability)

What it reports (no accuracy):
- Numeric stochasticity for PS4 counts:
    * Variant_found_consistent (per PMID; all runs agree on found/not found)
    * Count_variance
    * Pct_unstable_count = 1 - majority fraction of identical counts
    * Pct_under / Pct_exact / Pct_over (direction vs truth; context only)
- Variant_found distribution:
    * Pct_variant_found_yes / Pct_variant_found_no (per PMID)
- Semantic stochasticity:
    * Just_* and Reason_*: sem_mean/median/min and %>=0.8/%>=0.9
    * Consolidated semantic metrics (weighted: 0.6 reasoning, 0.4 justification)
- Composite Stability_index:
    = 0.30*(1 - Pct_unstable_count)
    + 0.20*Variant_found_consistent
    + 0.25*Reason_sem_mean
    + 0.25*Just_sem_mean
"""

import os, math, json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI

# ------------ CONFIG ------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

DEFAULT_INPUT_FILES = {
    "gpt5high":   str(INPUT_DIR / "v7_gpt5high_variant_summary__det_onlydetset_nostochastic.csv"),
    "claude":     str(INPUT_DIR / "v7_claude_variant_summary__det_detsetonly_nostochastic.csv"),
    "gemini":     str(INPUT_DIR / "v7_gemini_variant_summary__det_onlydetset_nostochastic.csv"),
    "o4minihigh": str(INPUT_DIR / "v7_o4minihigh_variant_summary__det_onlydetset_nostochastic.csv"),
    "o3":         str(INPUT_DIR / "v7_o3_variant_summary__det_onlydetset_nostochastic.csv"),
}

GROUP_COL = "PMID"
RUN_COL   = "run_index"

COUNT_COL = "total_ps4_case_counts"
FOUND_COL = "variant_found"  # boolean or {True/False}

TRUTH_COUNT_COL = "truth_total_ps4_case_counts"  # used only for direction (under/exact/over)
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_THRESHOLDS  = (0.80, 0.90)

# Consolidated semantics weight
CONSOLIDATED_WEIGHT_REASON = 0.6

# ------------ CLIENT ------------
def make_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=key)

# ------------ Cosine + Stats ------------
def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    emb = np.array(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return (emb @ emb.T) / (norms @ norms.T)

def offdiag_values(sim: np.ndarray) -> np.ndarray:
    if sim.shape[0] < 2:
        return np.array([1.0], dtype=np.float32)
    iu = np.triu_indices_from(sim, k=1)
    return sim[iu].astype(np.float32)

def aggregate_stats(offdiag: np.ndarray, thresholds=DEFAULT_THRESHOLDS) -> Dict[str, float]:
    offdiag = np.asarray(offdiag, dtype=np.float32)
    stats = {
        "sem_mean": float(np.mean(offdiag)),
        "sem_median": float(np.median(offdiag)),
        "sem_min": float(np.min(offdiag)),
    }
    for t in thresholds:
        stats[f"sem_prop_ge_{str(t).replace('.','_')}"] = float((offdiag >= t).mean())
    return stats

# ------------ Embedding cache ------------
class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.mem: Dict[str, List[float]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        self.mem[obj["text"]] = obj["embedding"]
                    except:
                        continue
    def get(self, text: str):
        return self.mem.get(text)
    def set(self, text: str, emb: List[float]):
        self.mem[text] = emb
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text, "embedding": emb}) + "\n")

# ------------ Embedding wrapper ------------
class Embedder:
    def __init__(self, client: OpenAI, model: str, cache: EmbeddingCache, batch_size=96):
        self.client = client
        self.model = model
        self.cache = cache
        self.batch_size = batch_size

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6))
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = [None] * len(texts)
        pending_idx, pending_vals = [], []
        for i, t in enumerate(texts):
            t_norm = (t or "").strip()
            if not t_norm:
                out[i] = np.zeros(1536, dtype=np.float32)
                continue
            cached = self.cache.get(t_norm)
            if cached is not None:
                out[i] = np.array(cached, dtype=np.float32)
            else:
                pending_idx.append(i)
                pending_vals.append(t_norm)
        for s in range(0, len(pending_vals), self.batch_size):
            chunk = pending_vals[s:s+self.batch_size]
            if not chunk: continue
            embs = self._embed_batch(chunk)
            for j, emb in enumerate(embs):
                vec = np.array(emb, dtype=np.float32)
                txt = chunk[j]
                self.cache.set(txt, emb)
                out[pending_idx[s+j]] = vec
        return out

# ------------ Semantic stats ------------
def semantic_similarity_stats(texts: List[str], embedder: Embedder, thresholds=DEFAULT_THRESHOLDS) -> Dict[str, float]:
    clean = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(clean) < 2:
        base = {"sem_mean": 1.0, "sem_median": 1.0, "sem_min": 1.0}
        for t in thresholds:
            base[f"sem_prop_ge_{str(t).replace('.','_')}"] = 1.0
        return base
    embs = embedder.embed_texts(clean)
    sim = cosine_matrix(embs)
    off = offdiag_values(sim)
    return aggregate_stats(off, thresholds)

# ------------ Core analysis ------------
def analyze_files(input_files, embed_model, thresholds, out_prefix):
    client = make_client()
    cache = EmbeddingCache(BASE_DIR / f"{out_prefix}_emb_cache.jsonl")
    embedder = Embedder(client, embed_model, cache)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, path in input_files.items():
        df = pd.read_csv(path)

        for pmid, grp in df.groupby(GROUP_COL):
            runs = len(grp)

            # Numeric stochasticity (PS4 counts)
            counts = grp[COUNT_COL].dropna().astype(float).tolist()
            count_var = float(np.var(counts)) if counts else float("nan")
            cnt_min, cnt_max = (float(np.min(counts)), float(np.max(counts))) if counts else (float("nan"), float("nan"))

            # % unstable runs = 1 - (majority fraction)
            if len(counts) > 0:
                counts_arr = np.array(counts)
                _, counts_freq = np.unique(counts_arr, return_counts=True)
                pct_majority = counts_freq.max() / len(counts_arr)
                pct_unstable = 1 - pct_majority
            else:
                pct_majority = float("nan")
                pct_unstable = float("nan")

            # Bias direction vs truth (context only)
            truth_count = grp[TRUTH_COUNT_COL].iloc[0] if TRUTH_COUNT_COL in grp.columns else float("nan")
            if pd.notna(truth_count) and len(counts) > 0:
                arr = np.array(counts)
                pct_under = float((arr < truth_count).mean())
                pct_exact = float((arr == truth_count).mean())
                pct_over  = float((arr > truth_count).mean())
            else:
                pct_under = pct_exact = pct_over = float("nan")

            # Variant_found stability + distribution
            founds = grp[FOUND_COL].dropna().tolist()
            variant_found_consistent = (len(set(map(str, founds))) == 1) if len(founds) else None
            if len(founds) > 0:
                # coerce truthy strings to bools where possible
                vals = [str(v).strip().lower() in ("1","true","yes","y","t") for v in founds]
                pct_found_yes = float(np.mean(vals))
                pct_found_no  = 1.0 - pct_found_yes
            else:
                pct_found_yes = pct_found_no = float("nan")

            # Semantic stats
            just_stats   = semantic_similarity_stats(grp.get("counting_justification", []), embedder, thresholds)
            reason_stats = semantic_similarity_stats(grp.get("model_reasoning", []), embedder, thresholds)

            # Consolidated semantics (weighted)
            sem_mean_consol   = CONSOLIDATED_WEIGHT_REASON * reason_stats["sem_mean"] + (1 - CONSOLIDATED_WEIGHT_REASON) * just_stats["sem_mean"]
            sem_median_consol = CONSOLIDATED_WEIGHT_REASON * reason_stats["sem_median"] + (1 - CONSOLIDATED_WEIGHT_REASON) * just_stats["sem_median"]
            sem_min_consol    = min(reason_stats["sem_min"], just_stats["sem_min"])
            cons_props = {}
            for t in thresholds:
                k = f"sem_prop_ge_{str(t).replace('.','_')}"
                cons_props[f"Semantic_prop_ge_{str(t).replace('.','_')}_consolidated"] = float((reason_stats[k] + just_stats[k]) / 2.0)

            # --- Stability index (counts + variant_found + semantics) ---
            ps4_count_stability = 1 - (pct_unstable if not math.isnan(pct_unstable) else 0.0)
            vf_stability = 1.0 if variant_found_consistent else 0.0 if variant_found_consistent is not None else 0.0
            stability_index = (
                0.30 * ps4_count_stability +
                0.20 * vf_stability +
                0.25 * reason_stats["sem_mean"] +
                0.25 * just_stats["sem_mean"]
            )

            row = {
                "Model": model_name, "PMID": pmid, "Runs": runs,

                # Numeric stochasticity (PS4 counts)
                "Count_variance": count_var,
                "Count_min": cnt_min, "Count_max": cnt_max,
                "Pct_majority_count": pct_majority,
                "Pct_unstable_count": pct_unstable,
                "Pct_under": pct_under, "Pct_exact": pct_exact, "Pct_over": pct_over,

                # Variant_found stability + distribution
                "Variant_found_consistent": variant_found_consistent,
                "Pct_variant_found_yes": pct_found_yes,
                "Pct_variant_found_no": pct_found_no,

                # Semantic (Justification)
                "Just_sem_mean": just_stats["sem_mean"],
                "Just_sem_median": just_stats["sem_median"],
                "Just_sem_min": just_stats["sem_min"],
                "Just_sem_prop_ge_0_8": just_stats.get("sem_prop_ge_0_8", np.nan),
                "Just_sem_prop_ge_0_9": just_stats.get("sem_prop_ge_0_9", np.nan),

                # Semantic (Reasoning)
                "Reason_sem_mean": reason_stats["sem_mean"],
                "Reason_sem_median": reason_stats["sem_median"],
                "Reason_sem_min": reason_stats["sem_min"],
                "Reason_sem_prop_ge_0_8": reason_stats.get("sem_prop_ge_0_8", np.nan),
                "Reason_sem_prop_ge_0_9": reason_stats.get("sem_prop_ge_0_9", np.nan),

                # Consolidated semantics
                "Semantic_mean_consolidated": sem_mean_consol,
                "Semantic_median_consolidated": sem_median_consol,
                "Semantic_min_consolidated": sem_min_consol,

                # Composite
                "Stability_index": stability_index,
            }
            row.update(cons_props)
            rows.append(row)

    res = pd.DataFrame(rows)

    # By-model summary (means across PMIDs)
    summary = (
        res.groupby("Model")
          .agg({
              "PMID": "nunique",
              "Variant_found_consistent": "mean",
              "Count_variance": lambda x: (pd.Series(x) > 0).mean(),  # % pubs with variable counts
              "Pct_unstable_count": "mean",
              "Pct_variant_found_yes": "mean",
              "Pct_variant_found_no": "mean",
              "Pct_under": "mean",
              "Pct_exact": "mean",
              "Pct_over": "mean",

              "Just_sem_mean": "mean",
              "Just_sem_median": "mean",
              "Just_sem_min": "mean",
              "Just_sem_prop_ge_0_8": "mean",
              "Just_sem_prop_ge_0_9": "mean",

              "Reason_sem_mean": "mean",
              "Reason_sem_median": "mean",
              "Reason_sem_min": "mean",
              "Reason_sem_prop_ge_0_8": "mean",
              "Reason_sem_prop_ge_0_9": "mean",

              "Semantic_mean_consolidated": "mean",
              "Semantic_median_consolidated": "mean",
              "Semantic_min_consolidated": "mean",
              "Semantic_prop_ge_0_8_consolidated": "mean",
              "Semantic_prop_ge_0_9_consolidated": "mean",

              "Stability_index": "mean",
          })
          .reset_index()
          .rename(columns={
              "PMID": "pubs",
              "Variant_found_consistent": "pct_variant_found_consistent",
              "Count_variance": "pct_variable_counts",
          })
    )

    res.to_csv(OUTPUT_DIR / "nostochasticity_only_per_pmid.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "nostochasticity_only_summary_by_model.csv", index=False)
    return res, summary

# ------------ Run ------------
if __name__ == "__main__":
    res, summary = analyze_files(DEFAULT_INPUT_FILES, DEFAULT_EMBED_MODEL, DEFAULT_THRESHOLDS, "ps4_nostochasticity")
    print(
        "Wrote:\n - "
        + str(OUTPUT_DIR / "nostochasticity_only_per_pmid.csv")
        + "\n - "
        + str(OUTPUT_DIR / "nostochasticity_only_summary_by_model.csv")
    )
    print(summary)



#sanity test - python nostochasticitychecker.py --test --model gpt5high --pmid 31342592 --plots

#full test - python nostochasticitychecker.py --plots
