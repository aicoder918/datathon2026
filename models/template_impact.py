"""OOF template-impact features.

For each unique headline template, compute the mean training return for
sessions in which that template appears (Bayesian-shrunk to the global mean).
Use the table to score new sessions by Σ template_impact[T] over their seen
headlines.

Computed strictly per-fold inside CV: the val/test sessions never see their
own r when the impact table is built. For final inference, the table is built
on all 1000 training sessions.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

from models.features import SEEN_MAX, LATE_START

DOLLAR_RE = re.compile(r"\$\d+(?:\.\d+)?(?:[BMK])?")
PCT_RE = re.compile(r"\d+(?:\.\d+)?%")
NUM_RE = re.compile(r"\b\d+\b")


def extract_template(headline: str) -> str:
    """Strip the company name (first 2 words) and numerals -> structural template."""
    if not isinstance(headline, str):
        return ""
    parts = headline.split()
    if len(parts) < 2:
        return headline
    rest = " ".join(parts[2:])
    rest = DOLLAR_RE.sub("<$>", rest)
    rest = PCT_RE.sub("<P>", rest)
    rest = NUM_RE.sub("<N>", rest)
    return rest


def attach_template(headlines: pd.DataFrame) -> pd.DataFrame:
    h = headlines[headlines["bar_ix"] <= SEEN_MAX].copy()
    h["template"] = h["headline"].apply(extract_template)
    return h


def compute_impact_table(
    headlines_with_tpl: pd.DataFrame,
    r_train: pd.Series,
    alpha: float = 30.0,
    bar_ix_min: int | None = None,
) -> pd.Series:
    """Return template -> shrunk mean(r) over training sessions where template appears.

    Bayesian shrinkage to the global mean with strength `alpha`:
        impact[T] = (n_T * mean_r_T + alpha * global_mean) / (n_T + alpha)

    Importantly: each (template, session) pair counts ONCE — we don't
    double-weight sessions that have the same template more than once.
    """
    h = headlines_with_tpl
    if bar_ix_min is not None:
        h = h[h["bar_ix"] >= bar_ix_min]
    r_train = r_train.copy()
    r_train.name = "r"
    pairs = (h[["session", "template"]]
             .drop_duplicates()
             .merge(r_train, left_on="session", right_index=True, how="inner"))
    global_mean = float(r_train.mean())
    agg = pairs.groupby("template").agg(n=("r", "size"), sum_r=("r", "sum"))
    agg["impact"] = (agg["sum_r"] + alpha * global_mean) / (agg["n"] + alpha)
    return agg["impact"]


def add_template_scores(
    X: pd.DataFrame,
    headlines_with_tpl: pd.DataFrame,
    impact_late: pd.Series,
    impact_all: pd.Series,
) -> pd.DataFrame:
    """Add 'tpl_late_score' and 'tpl_all_score' columns to X (in-place return).

    tpl_late_score(session) = Σ impact_late[T(h)] over headlines in bars 40..49
    tpl_all_score(session)  = Σ impact_all[T(h)] over headlines in bars 0..49

    Templates not seen in training fall back to the global mean from the impact
    table (the impact value with the most shrinkage = global mean ± a tiny bit).
    """
    h = headlines_with_tpl

    fallback_late = float(impact_late.mean()) if len(impact_late) else 0.0
    fallback_all = float(impact_all.mean()) if len(impact_all) else 0.0

    h_all = h.assign(impact=h["template"].map(impact_all).fillna(fallback_all))
    h_late = h[h["bar_ix"] >= LATE_START].assign(
        impact=h.loc[h["bar_ix"] >= LATE_START, "template"]
                 .map(impact_late).fillna(fallback_late)
    )

    s_all = h_all.groupby("session")["impact"].sum().reindex(X.index, fill_value=fallback_all)
    s_late = h_late.groupby("session")["impact"].sum().reindex(X.index, fill_value=fallback_late)

    X = X.copy()
    X["tpl_all_score"] = s_all.values
    X["tpl_late_score"] = s_late.values
    return X
