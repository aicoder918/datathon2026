"""Outer CV with OOF template-impact features.

Per-fold pipeline:
  1. Compute impact_all and impact_late from *training-fold sessions only*.
  2. Fill tpl_all_score / tpl_late_score columns on X_tr and X_va using those tables.
  3. Fit MeanHead (Ridge) + VarianceHead on X_tr, predict X_va.
  4. Pool OOF -> Sharpe + bootstrap CI.

Fixed hyperparameters (no inner CV to avoid the v2 selection mistake):
  alpha=100, lam=1.0, gamma=0.006 (v1's settings).

Two comparison variants:
  - base:  PRICE+HEADLINE only (13 stable features)
  - full:  PRICE+HEADLINE + template features
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.features import (
    DATA_DIR, FEATURE_COLS, PRICE_FEATURES, HEADLINE_FEATURES, TEMPLATE_FEATURES,
    LATE_START, build_features, compute_train_target, load_finbert,
)
from models.sharpe_model import (
    MeanHead, VarianceHead, positions_from_heads, sharpe, STABLE_FEATURES,
)
from models.cv_evaluate import bootstrap_sharpe
from models.template_impact import (
    attach_template, compute_impact_table, add_template_scores,
)

OUTER = 10
ALPHA_M = 100.0
ALPHA_V = 5.0
LAM = 1.0
GAMMA = 0.006
SHRINK_ALPHA = 30.0  # shrinkage strength for template target encoding


def _run_outer(feature_cols: list, X_base: pd.DataFrame, r: np.ndarray,
               sessions: np.ndarray, headlines_with_tpl: pd.DataFrame,
               use_template: bool, name: str) -> dict:
    gkf = GroupKFold(n_splits=OUTER)
    mu_oof = np.zeros(len(X_base))
    s2_oof = np.zeros(len(X_base))
    ref_parts = []

    r_series_full = pd.Series(r, index=X_base.index, name="r")

    for tr_idx, va_idx in gkf.split(X_base, groups=sessions):
        X_tr = X_base.iloc[tr_idx].copy()
        X_va = X_base.iloc[va_idx].copy()
        r_tr = r[tr_idx]

        if use_template:
            # Impact tables computed on training fold only
            train_sessions = X_tr.index
            r_tr_ser = r_series_full.loc[train_sessions]
            h_train = headlines_with_tpl[
                headlines_with_tpl["session"].isin(train_sessions)
            ]
            impact_all = compute_impact_table(h_train, r_tr_ser,
                                              alpha=SHRINK_ALPHA)
            impact_late = compute_impact_table(h_train, r_tr_ser,
                                               alpha=SHRINK_ALPHA,
                                               bar_ix_min=LATE_START)
            # Apply (train fold's headlines for X_tr, all sessions' visible headlines for X_va)
            X_tr = add_template_scores(X_tr, h_train, impact_late, impact_all)
            h_val = headlines_with_tpl[
                headlines_with_tpl["session"].isin(X_va.index)
            ]
            X_va = add_template_scores(X_va, h_val, impact_late, impact_all)

        mh = MeanHead(alpha=ALPHA_M, features=list(feature_cols)).fit(X_tr, r_tr)
        vh = VarianceHead(alpha=ALPHA_V).fit(X_tr, r_tr)

        mu_va = mh.predict(X_va); s2_va = vh.predict(X_va)
        mu_tr = mh.predict(X_tr); s2_tr = vh.predict(X_tr)
        mu_oof[va_idx] = mu_va
        s2_oof[va_idx] = s2_va
        ref_parts.append((mu_tr + GAMMA) / np.maximum(s2_tr, 1e-12))

    ref_raw = np.concatenate(ref_parts)

    p = positions_from_heads(mu_oof, s2_oof, gamma=GAMMA, lam=LAM, ref_raw=ref_raw)
    sh, lo, hi, se = bootstrap_sharpe(p * r)
    corr_mu = float(np.corrcoef(mu_oof, r)[0, 1])
    corr_s2 = float(np.corrcoef(s2_oof, r * r)[0, 1])
    return {
        "name": name, "n_features": len(feature_cols),
        "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "se": se,
        "corr_mu_r": corr_mu, "corr_s2_r2": corr_s2,
        "p_mean": float(p.mean()), "p_std": float(p.std()),
        "p_min": float(p.min()), "p_max": float(p.max()),
    }


def main():
    bs = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    bu = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    hs = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
    fb = load_finbert()

    X = build_features(bs, hs, finbert=fb)
    r_ser = compute_train_target(bs, bu).reindex(X.index)
    r = r_ser.to_numpy()
    sessions = X.index.to_numpy()

    h_tpl = attach_template(hs)  # adds "template" column for seen headlines
    print(f"n_sessions={len(X)}  unique_templates={h_tpl['template'].nunique()}  "
          f"n_headlines={len(h_tpl)}")

    # Baseline: constant-long
    base_pnl = 1.0 * r
    bs_sh, bs_lo, bs_hi, bs_se = bootstrap_sharpe(base_pnl)
    print(f"Baseline constant_long: sharpe={bs_sh:.3f}  "
          f"CI=[{bs_lo:.3f},{bs_hi:.3f}]  SE={bs_se:.3f}")
    print()

    variants = [
        # Base: 13 stable features (v1 equivalent)
        ("v1_stable13", list(STABLE_FEATURES), False),
        # Stable + templates
        ("v3_stable_tpl", list(STABLE_FEATURES) + list(TEMPLATE_FEATURES), True),
        # All price+headline (22) + templates (24)
        ("v3_all_tpl", list(PRICE_FEATURES) + list(HEADLINE_FEATURES) + list(TEMPLATE_FEATURES), True),
        # All price+headline, no templates (22 features)
        ("v3_all_notpl", list(PRICE_FEATURES) + list(HEADLINE_FEATURES), False),
    ]

    rows = []
    for name, feats, use_tpl in variants:
        r_ = _run_outer(feats, X, r, sessions, h_tpl, use_tpl, name)
        print(f"  {name} ({r_['n_features']} feats): "
              f"sharpe={r_['sharpe']:.3f}  CI=[{r_['ci_lo']:.3f},{r_['ci_hi']:.3f}]  "
              f"delta={r_['sharpe']-bs_sh:+.3f} ({(r_['sharpe']-bs_sh)/bs_se:+.2f} SE)  "
              f"corr(mu,r)={r_['corr_mu_r']:+.4f}  "
              f"p_std={r_['p_std']:.3f}  p=[{r_['p_min']:.2f},{r_['p_max']:.2f}]")
        rows.append(r_)

    print()
    print("=== Summary ===")
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df["delta_vs_const"] = df["sharpe"] - bs_sh
    df["delta_in_se"] = df["delta_vs_const"] / bs_se
    print(df[["name", "n_features", "sharpe", "ci_lo", "ci_hi",
              "delta_vs_const", "delta_in_se",
              "corr_mu_r", "p_std"]].to_string(index=False,
                                               float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
