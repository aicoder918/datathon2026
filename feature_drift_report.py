"""Train-vs-test feature diagnostics for model-input DataFrames.

This module is designed to be imported from notebooks, e.g.:

    from feature_drift_report import (
        compare_train_test_features,
        summarize_drift,
        save_feature_drift_outputs,
        plot_top_drift_features,
    )

    report = compare_train_test_features(X_train, X_test)
    summary = summarize_drift(report)
    save_feature_drift_outputs(report, summary, out_dir="plots/feature_drift")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12
DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _to_numeric_values(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _basic_stats(values: np.ndarray, quantiles: Sequence[float]) -> dict[str, float]:
    if values.size == 0:
        out = {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "zero_rate": float("nan"),
        }
        for q in quantiles:
            out[f"q{int(round(q * 100)):02d}"] = float("nan")
        out["iqr"] = float("nan")
        return out

    out = {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "zero_rate": float(np.mean(values == 0)),
    }
    for q in quantiles:
        out[f"q{int(round(q * 100)):02d}"] = _safe_quantile(values, q)
    q75 = _safe_quantile(values, 0.75)
    q25 = _safe_quantile(values, 0.25)
    out["iqr"] = float(q75 - q25) if np.isfinite(q75) and np.isfinite(q25) else float("nan")
    return out


def _compute_psi(train_vals: np.ndarray, test_vals: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index (PSI): higher means stronger shift."""
    if train_vals.size == 0 or test_vals.size == 0:
        return float("nan")

    # Quantile bins on train; collapse duplicate edges if necessary.
    edges = np.quantile(train_vals, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)

    if edges.size < 3:
        lo = float(min(np.min(train_vals), np.min(test_vals)))
        hi = float(max(np.max(train_vals), np.max(test_vals)))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            # Both degenerate and equal -> no distribution shift detected.
            return 0.0
        edges = np.linspace(lo, hi, n_bins + 1)

    train_hist, _ = np.histogram(train_vals, bins=edges)
    test_hist, _ = np.histogram(test_vals, bins=edges)

    train_pct = np.clip(train_hist / max(train_hist.sum(), 1), EPS, None)
    test_pct = np.clip(test_hist / max(test_hist.sum(), 1), EPS, None)
    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
    return float(psi)


def _ks_and_wasserstein(train_vals: np.ndarray, test_vals: np.ndarray) -> tuple[float, float, float]:
    if train_vals.size < 2 or test_vals.size < 2:
        return float("nan"), float("nan"), float("nan")
    ks = stats.ks_2samp(train_vals, test_vals, alternative="two-sided", mode="auto")
    wass = stats.wasserstein_distance(train_vals, test_vals)
    return float(ks.statistic), float(ks.pvalue), float(wass)


def _resolve_feature_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str] | None,
    numeric_only: bool,
) -> list[str]:
    if feature_cols is None:
        cols = sorted(set(train_df.columns).intersection(test_df.columns))
    else:
        cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

    if not numeric_only:
        return cols

    keep: list[str] = []
    for c in cols:
        # Keep if either side is numeric-like to avoid dropping sparse numeric cols.
        is_num_train = pd.api.types.is_numeric_dtype(train_df[c])
        is_num_test = pd.api.types.is_numeric_dtype(test_df[c])
        if is_num_train or is_num_test:
            keep.append(c)
    return keep


def compare_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    *,
    numeric_only: bool = True,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    psi_bins: int = 10,
    psi_warn: float = 0.20,
    ks_warn: float = 0.20,
    ks_pvalue_warn: float = 0.01,
    mean_shift_warn_sigma: float = 1.0,
) -> pd.DataFrame:
    """Compare per-feature distributions between train and test.

    Returns one row per feature with descriptive statistics and drift metrics.
    """
    cols = _resolve_feature_cols(train_df, test_df, feature_cols, numeric_only=numeric_only)
    rows: list[dict[str, float | str | bool]] = []

    n_train_rows = float(len(train_df))
    n_test_rows = float(len(test_df))

    for col in cols:
        tr_raw = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)
        te_raw = pd.to_numeric(test_df[col], errors="coerce").to_numpy(dtype=float)
        tr_vals = tr_raw[np.isfinite(tr_raw)]
        te_vals = te_raw[np.isfinite(te_raw)]

        tr_stats = _basic_stats(tr_vals, quantiles)
        te_stats = _basic_stats(te_vals, quantiles)

        ks_stat, ks_pvalue, wass = _ks_and_wasserstein(tr_vals, te_vals)
        psi = _compute_psi(tr_vals, te_vals, n_bins=psi_bins)

        tr_mean = tr_stats["mean"]
        te_mean = te_stats["mean"]
        tr_std = tr_stats["std"]
        tr_median = tr_stats["q50"]
        te_median = te_stats["q50"]
        tr_iqr = tr_stats["iqr"]
        te_iqr = te_stats["iqr"]

        mean_shift_sigma = (te_mean - tr_mean) / (tr_std + EPS) if np.isfinite(tr_mean) and np.isfinite(te_mean) else float("nan")
        abs_mean_shift_sigma = abs(mean_shift_sigma) if np.isfinite(mean_shift_sigma) else float("nan")
        median_shift_iqr = (te_median - tr_median) / (tr_iqr + EPS) if np.isfinite(tr_median) and np.isfinite(te_median) else float("nan")
        abs_median_shift_iqr = abs(median_shift_iqr) if np.isfinite(median_shift_iqr) else float("nan")
        std_ratio = te_stats["std"] / (tr_stats["std"] + EPS) if np.isfinite(te_stats["std"]) and np.isfinite(tr_stats["std"]) else float("nan")
        iqr_ratio = te_iqr / (tr_iqr + EPS) if np.isfinite(te_iqr) and np.isfinite(tr_iqr) else float("nan")

        drift_reasons: list[str] = []
        if np.isfinite(psi) and psi >= psi_warn:
            drift_reasons.append(f"psi>={psi_warn}")
        if np.isfinite(ks_stat) and np.isfinite(ks_pvalue) and ks_stat >= ks_warn and ks_pvalue <= ks_pvalue_warn:
            drift_reasons.append(f"ks>={ks_warn}&p<={ks_pvalue_warn}")
        if np.isfinite(abs_mean_shift_sigma) and abs_mean_shift_sigma >= mean_shift_warn_sigma:
            drift_reasons.append(f"|mean_shift|>={mean_shift_warn_sigma}sigma")

        drift_flag = len(drift_reasons) > 0
        drift_score = (
            np.nan_to_num(abs_mean_shift_sigma, nan=0.0)
            + np.nan_to_num(abs_median_shift_iqr, nan=0.0)
            + np.nan_to_num(psi, nan=0.0)
            + np.nan_to_num(ks_stat, nan=0.0)
        )

        row: dict[str, float | str | bool] = {
            "feature": col,
            "dtype_train": str(train_df[col].dtype),
            "dtype_test": str(test_df[col].dtype),
            "n_train_rows": n_train_rows,
            "n_test_rows": n_test_rows,
            "n_train_nonnull": tr_stats["count"],
            "n_test_nonnull": te_stats["count"],
            "train_nonnull_rate": tr_stats["count"] / max(n_train_rows, 1.0),
            "test_nonnull_rate": te_stats["count"] / max(n_test_rows, 1.0),
            "train_missing_rate": 1.0 - tr_stats["count"] / max(n_train_rows, 1.0),
            "test_missing_rate": 1.0 - te_stats["count"] / max(n_test_rows, 1.0),
            "train_nunique": float(pd.Series(tr_vals).nunique()),
            "test_nunique": float(pd.Series(te_vals).nunique()),
            "train_is_constant": bool(pd.Series(tr_vals).nunique() <= 1),
            "test_is_constant": bool(pd.Series(te_vals).nunique() <= 1),
            "train_zero_rate": tr_stats["zero_rate"],
            "test_zero_rate": te_stats["zero_rate"],
            "train_mean": tr_mean,
            "test_mean": te_mean,
            "train_std": tr_std,
            "test_std": te_stats["std"],
            "train_min": tr_stats["min"],
            "test_min": te_stats["min"],
            "train_max": tr_stats["max"],
            "test_max": te_stats["max"],
            "mean_shift_sigma": mean_shift_sigma,
            "abs_mean_shift_sigma": abs_mean_shift_sigma,
            "median_shift_iqr": median_shift_iqr,
            "abs_median_shift_iqr": abs_median_shift_iqr,
            "std_ratio_test_over_train": std_ratio,
            "iqr_ratio_test_over_train": iqr_ratio,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "wasserstein": wass,
            "psi": psi,
            "drift_flag": drift_flag,
            "drift_reasons": ",".join(drift_reasons),
            "drift_score": drift_score,
        }

        for q in quantiles:
            q_name = f"q{int(round(q * 100)):02d}"
            row[f"train_{q_name}"] = tr_stats[q_name]
            row[f"test_{q_name}"] = te_stats[q_name]

        rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["drift_flag", "drift_score"], ascending=[False, False]).reset_index(drop=True)
    return report


def summarize_drift(
    report_df: pd.DataFrame,
    *,
    psi_warn: float = 0.20,
    ks_warn: float = 0.20,
    ks_pvalue_warn: float = 0.01,
    mean_shift_warn_sigma: float = 1.0,
    top_k: int = 20,
) -> dict[str, object]:
    """Aggregate summary for a report generated by compare_train_test_features."""
    if report_df.empty:
        return {
            "n_features": 0,
            "n_flagged": 0,
            "flagged_rate": 0.0,
            "n_psi_high": 0,
            "n_ks_high": 0,
            "n_mean_shift_high": 0,
            "top_features_by_drift_score": [],
            "thresholds": {
                "psi_warn": psi_warn,
                "ks_warn": ks_warn,
                "ks_pvalue_warn": ks_pvalue_warn,
                "mean_shift_warn_sigma": mean_shift_warn_sigma,
            },
        }

    psi_high = report_df["psi"] >= psi_warn
    ks_high = (report_df["ks_stat"] >= ks_warn) & (report_df["ks_pvalue"] <= ks_pvalue_warn)
    mean_high = report_df["abs_mean_shift_sigma"] >= mean_shift_warn_sigma
    flagged = report_df["drift_flag"].fillna(False)

    top_df = report_df.sort_values("drift_score", ascending=False).head(top_k)
    top_features = top_df[
        ["feature", "drift_score", "psi", "ks_stat", "ks_pvalue", "abs_mean_shift_sigma", "drift_reasons"]
    ].to_dict(orient="records")

    return {
        "n_features": int(len(report_df)),
        "n_flagged": int(flagged.sum()),
        "flagged_rate": float(flagged.mean()),
        "n_psi_high": int(psi_high.fillna(False).sum()),
        "n_ks_high": int(ks_high.fillna(False).sum()),
        "n_mean_shift_high": int(mean_high.fillna(False).sum()),
        "top_features_by_drift_score": top_features,
        "thresholds": {
            "psi_warn": psi_warn,
            "ks_warn": ks_warn,
            "ks_pvalue_warn": ks_pvalue_warn,
            "mean_shift_warn_sigma": mean_shift_warn_sigma,
        },
    }


def save_feature_drift_outputs(
    report_df: pd.DataFrame,
    summary: dict[str, object],
    out_dir: str | Path,
    *,
    prefix: str = "feature_drift",
) -> dict[str, Path]:
    """Save report CSV and summary JSON to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_path = out_path / f"{prefix}_report.csv"
    summary_path = out_path / f"{prefix}_summary.json"

    report_df.to_csv(report_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    return {"report_csv": report_path, "summary_json": summary_path}


def plot_feature_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature: str,
    *,
    bins: int = 50,
    density: bool = True,
    ax=None,
):
    """Overlay train/test histogram for a single feature."""
    import matplotlib.pyplot as plt

    tr_vals = _to_numeric_values(train_df[feature])
    te_vals = _to_numeric_values(test_df[feature])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.hist(tr_vals, bins=bins, density=density, alpha=0.45, label="train")
    ax.hist(te_vals, bins=bins, density=density, alpha=0.45, label="test")
    ax.set_title(feature)
    ax.set_xlabel("value")
    ax.set_ylabel("density" if density else "count")
    ax.legend()
    return ax


def plot_top_drift_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report_df: pd.DataFrame,
    *,
    top_k: int = 12,
    bins: int = 40,
    out_path: str | Path | None = None,
):
    """Plot overlaid histograms for highest-drift features from report_df."""
    import matplotlib.pyplot as plt

    if report_df.empty:
        raise ValueError("report_df is empty")

    top_features = report_df.sort_values("drift_score", ascending=False)["feature"].head(top_k).tolist()
    n = len(top_features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(top_features):
        plot_feature_distribution(train_df, test_df, feat, bins=bins, density=True, ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig

