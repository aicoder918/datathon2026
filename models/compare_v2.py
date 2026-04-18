"""One-shot comparison of v2 mean-head variants.

Runs the same outer 10-fold GroupKFold for each of:
  - Ridge on all 22 features
  - Ridge on stable subset (13 features)
  - CatBoost on all 22 features
  - CatBoost on stable subset (13 features)

Inner-CV-free: uses fixed (alpha, gamma, lam=1.0) to keep the comparison apples-to-apples
and to surface raw mean-head capacity. We sweep lam at the end on pooled OOF.

Pooled OOF Sharpe + bootstrap CI per variant; baseline-comparison table.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.features import (
    DATA_DIR, FEATURE_COLS, build_features, compute_train_target, load_finbert,
)
from models.sharpe_model import (
    MeanHead, CatMeanHead, VarianceHead, positions_from_heads, sharpe,
    STABLE_FEATURES,
)
from models.cv_evaluate import bootstrap_sharpe

OUTER = 10
SEED = 42
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMA_GRID = [0.0, 0.002, 0.004, 0.006]


def _eval_variant(name: str, mean_factory: Callable, X: pd.DataFrame, r: np.ndarray,
                  sessions: np.ndarray) -> dict:
    gkf = GroupKFold(n_splits=OUTER)
    mu_oof = np.zeros(len(X))
    s2_oof = np.zeros(len(X))
    ref_parts = []
    for tr_idx, va_idx in gkf.split(X, groups=sessions):
        X_tr = X.iloc[tr_idx]; r_tr = r[tr_idx]
        X_va = X.iloc[va_idx]
        mh = mean_factory().fit(X_tr, r_tr)
        vh = VarianceHead(alpha=5.0).fit(X_tr, r_tr)
        mu_va = mh.predict(X_va); s2_va = vh.predict(X_va)
        mu_tr = mh.predict(X_tr); s2_tr = vh.predict(X_tr)
        mu_oof[va_idx] = mu_va; s2_oof[va_idx] = s2_va
        ref_parts.append((mu_tr + 0.0) / np.maximum(s2_tr, 1e-12))
    ref_raw = np.concatenate(ref_parts)

    # Sweep (gamma, lam) on pooled OOF
    rows = []
    for lam in LAMBDAS:
        best = {"gamma": 0.0, "sharpe": -1e9}
        for gamma in GAMMA_GRID:
            p = positions_from_heads(mu_oof, s2_oof, gamma=gamma, lam=lam,
                                     ref_raw=ref_raw)
            sh = sharpe(p * r)
            if sh > best["sharpe"]:
                best = {"gamma": gamma, "sharpe": sh}
        p = positions_from_heads(mu_oof, s2_oof, gamma=best["gamma"], lam=lam,
                                 ref_raw=ref_raw)
        sh, lo, hi, se = bootstrap_sharpe(p * r)
        rows.append({"variant": name, "lam": lam, "gamma": best["gamma"],
                     "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "se": se})
    df = pd.DataFrame(rows)
    df["corr_mu_r"] = float(np.corrcoef(mu_oof, r)[0, 1])
    df["corr_s2_r2"] = float(np.corrcoef(s2_oof, r * r)[0, 1])
    return {"table": df, "mu_oof": mu_oof, "s2_oof": s2_oof, "ref_raw": ref_raw}


def main():
    bs = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    bu = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    hs = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
    fb = load_finbert()
    X = build_features(bs, hs, finbert=fb)
    r_ser = compute_train_target(bs, bu).reindex(X.index)
    r = r_ser.to_numpy()
    sessions = X.index.to_numpy()

    print(f"n_sessions={len(X)}  n_features_all={len(FEATURE_COLS)}  "
          f"n_features_stable={len(STABLE_FEATURES)}")
    print(f"target: mean={r.mean():.5f}  std={r.std():.5f}")
    print()

    variants = [
        ("ridge_all_a100", lambda: MeanHead(alpha=100.0)),
        ("ridge_all_a300", lambda: MeanHead(alpha=300.0)),
        ("ridge_stable_a100", lambda: MeanHead(alpha=100.0, features=list(STABLE_FEATURES))),
        ("ridge_stable_a30", lambda: MeanHead(alpha=30.0, features=list(STABLE_FEATURES))),
        ("cat_all_d3", lambda: CatMeanHead(depth=3, iterations=300, l2_leaf_reg=10.0)),
        ("cat_stable_d3", lambda: CatMeanHead(depth=3, iterations=300, l2_leaf_reg=10.0,
                                              features=list(STABLE_FEATURES))),
        ("cat_stable_d2", lambda: CatMeanHead(depth=2, iterations=400, l2_leaf_reg=20.0,
                                              features=list(STABLE_FEATURES))),
    ]

    # Baseline rows
    baseline_pnl = 1.0 * r
    bs_sh, bs_lo, bs_hi, bs_se = bootstrap_sharpe(baseline_pnl)
    print(f"Baseline constant_long: sharpe={bs_sh:.3f}  CI=[{bs_lo:.3f},{bs_hi:.3f}]  SE={bs_se:.3f}")
    print()

    summary_rows = []
    for name, factory in variants:
        out = _eval_variant(name, factory, X, r, sessions)
        tbl = out["table"]
        best_row = tbl.sort_values("sharpe", ascending=False).iloc[0]
        summary_rows.append({
            "variant": name,
            "best_lam": best_row["lam"],
            "best_gamma": best_row["gamma"],
            "best_sharpe": best_row["sharpe"],
            "delta_vs_const": best_row["sharpe"] - bs_sh,
            "delta_in_se": (best_row["sharpe"] - bs_sh) / bs_se,
            "corr_mu_r": tbl["corr_mu_r"].iloc[0],
            "corr_s2_r2": tbl["corr_s2_r2"].iloc[0],
        })
        print(f"  {name}:  best_sharpe={best_row['sharpe']:.3f} "
              f"(lam={best_row['lam']}, gamma={best_row['gamma']:.4f})  "
              f"corr(mu,r)={tbl['corr_mu_r'].iloc[0]:+.4f}  "
              f"corr(s2,r2)={tbl['corr_s2_r2'].iloc[0]:+.4f}")

    print()
    print("=== Summary (sorted by best_sharpe) ===")
    summary = pd.DataFrame(summary_rows).sort_values("best_sharpe", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
