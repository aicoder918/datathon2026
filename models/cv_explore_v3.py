"""Exploratory CV: try interaction features and richer feature sets.

All variants fix {alpha=100, lam=1.0, gamma=0.006} so we measure *feature* quality,
not hyperparameter luck. N_OUTER=10, pooled OOF Sharpe + bootstrap CI.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.features import (
    DATA_DIR, PRICE_FEATURES, HEADLINE_FEATURES,
    build_features, compute_train_target, load_finbert,
)
from models.sharpe_model import (
    MeanHead, VarianceHead, positions_from_heads, sharpe, STABLE_FEATURES,
)
from models.cv_evaluate import bootstrap_sharpe

OUTER = 10
ALPHA_M = 100.0
ALPHA_V = 5.0
LAM = 1.0
GAMMA = 0.006


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features motivated by 'signal lives in bars 40-49'.

    Intuition:
    - Late news in a high-vol regime has a bigger absolute impact (vol regime gate).
    - Negative news on a drawdown should hit harder (bad-news amplification).
    - Late momentum * vol = "energy" proxy.
    """
    X = X.copy()
    # Standardize scale before multiplying (so products aren't dominated by variance)
    def _z(s):
        m = s.mean(); sd = s.std()
        return (s - m) / (sd + 1e-12)

    z_sent10 = _z(X["sent_last10"])
    z_sent_neg = _z(X["sent_neg_sum"])
    z_rv = _z(X["rv"])
    z_rv_last10 = _z(X["rv_last10"])
    z_pk = _z(X["pk_vol"])
    z_draw = _z(X["drawdown"])
    z_n = _z(X["n_headlines"])
    z_retlast5 = _z(X["ret_last5"])

    X["ix_sent10_rv"] = z_sent10 * z_rv
    X["ix_sent10_rvlast10"] = z_sent10 * z_rv_last10
    X["ix_sent10_draw"] = z_sent10 * z_draw
    X["ix_neg_rv"] = z_sent_neg * z_rv
    X["ix_ret5_rv"] = z_retlast5 * z_rv
    X["ix_n_rv"] = z_n * z_pk
    return X


INTERACTION_FEATURES = [
    "ix_sent10_rv", "ix_sent10_rvlast10", "ix_sent10_draw",
    "ix_neg_rv", "ix_ret5_rv", "ix_n_rv",
]


def _run(X: pd.DataFrame, r: np.ndarray, sessions: np.ndarray,
         feature_cols: list, name: str) -> dict:
    gkf = GroupKFold(n_splits=OUTER)
    mu_oof = np.zeros(len(X))
    s2_oof = np.zeros(len(X))
    ref_parts = []
    for tr_idx, va_idx in gkf.split(X, groups=sessions):
        X_tr = X.iloc[tr_idx]; r_tr = r[tr_idx]
        X_va = X.iloc[va_idx]
        mh = MeanHead(alpha=ALPHA_M, features=list(feature_cols)).fit(X_tr, r_tr)
        vh = VarianceHead(alpha=ALPHA_V).fit(X_tr, r_tr)
        mu_va = mh.predict(X_va); s2_va = vh.predict(X_va)
        mu_tr = mh.predict(X_tr); s2_tr = vh.predict(X_tr)
        mu_oof[va_idx] = mu_va; s2_oof[va_idx] = s2_va
        ref_parts.append((mu_tr + GAMMA) / np.maximum(s2_tr, 1e-12))
    ref_raw = np.concatenate(ref_parts)
    p = positions_from_heads(mu_oof, s2_oof, gamma=GAMMA, lam=LAM, ref_raw=ref_raw)
    sh, lo, hi, se = bootstrap_sharpe(p * r)
    return {
        "name": name, "n": len(feature_cols),
        "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "se": se,
        "corr_mu_r": float(np.corrcoef(mu_oof, r)[0, 1]),
        "p_std": float(p.std()), "p_min": float(p.min()), "p_max": float(p.max()),
        "mu_oof": mu_oof, "s2_oof": s2_oof, "ref_raw": ref_raw,
    }


def main():
    bs = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    bu = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    hs = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
    fb = load_finbert()

    X = build_features(bs, hs, finbert=fb)
    X = X.drop(columns=["tpl_late_score", "tpl_all_score"])  # drop placeholder zeros
    X_aug = add_interactions(X)
    r_ser = compute_train_target(bs, bu).reindex(X.index)
    r = r_ser.to_numpy()
    sessions = X.index.to_numpy()

    base_pnl = 1.0 * r
    bs_sh, bs_lo, bs_hi, bs_se = bootstrap_sharpe(base_pnl)
    print(f"constant_long: sharpe={bs_sh:.3f}  CI=[{bs_lo:.3f},{bs_hi:.3f}]  SE={bs_se:.3f}")
    print()

    stable = list(STABLE_FEATURES)
    all22 = list(PRICE_FEATURES) + list(HEADLINE_FEATURES)

    variants = [
        ("v1_stable13", X, stable),
        ("all22", X, all22),
        ("stable13+ix6", X_aug, stable + INTERACTION_FEATURES),
        ("all22+ix6", X_aug, all22 + INTERACTION_FEATURES),
        # Focused late-regime feature set
        ("late_focus", X, [
            "ret_last5", "ret_last10", "ret_last20",
            "rv", "rv_last10", "pk_vol",
            "drawdown",
            "sent_last10", "sent_neg_sum", "neg_frac", "n_headlines",
            "last_hl_signed", "last_hl_age",
        ]),
        ("late_focus+ix", X_aug, [
            "ret_last5", "ret_last10", "ret_last20",
            "rv", "rv_last10", "pk_vol", "drawdown",
            "sent_last10", "sent_neg_sum", "neg_frac", "n_headlines",
            "last_hl_signed", "last_hl_age",
            "ix_sent10_rv", "ix_sent10_draw", "ix_neg_rv",
        ]),
    ]

    results = []
    for name, Xi, feats in variants:
        r_ = _run(Xi, r, sessions, feats, name)
        print(f"  {name:<20} ({r_['n']:>2} feats): sharpe={r_['sharpe']:.3f}  "
              f"delta={r_['sharpe']-bs_sh:+.3f}  "
              f"corr(mu,r)={r_['corr_mu_r']:+.4f}  "
              f"p_std={r_['p_std']:.3f}  p=[{r_['p_min']:.2f},{r_['p_max']:.2f}]")
        results.append(r_)

    print()
    print("=== Sorted by sharpe ===")
    df = pd.DataFrame([{k: v for k, v in r_.items() if k not in ("mu_oof","s2_oof","ref_raw")}
                       for r_ in results]).sort_values("sharpe", ascending=False)
    df["delta_vs_const"] = df["sharpe"] - bs_sh
    df["delta_in_se"] = df["delta_vs_const"] / bs_se
    print(df[["name","n","sharpe","ci_lo","ci_hi","delta_vs_const","delta_in_se","corr_mu_r","p_std"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Ensemble: average positions from top-2 variants
    print()
    print("=== Ensemble (equal-weight positions, top-2) ===")
    # Take the top two
    ordered = sorted(results, key=lambda r_: r_["sharpe"], reverse=True)
    ra, rb = ordered[0], ordered[1]
    pa = positions_from_heads(ra["mu_oof"], ra["s2_oof"], gamma=GAMMA, lam=LAM, ref_raw=ra["ref_raw"])
    pb = positions_from_heads(rb["mu_oof"], rb["s2_oof"], gamma=GAMMA, lam=LAM, ref_raw=rb["ref_raw"])
    p_ens = 0.5 * (pa + pb)
    sh, lo, hi, _ = bootstrap_sharpe(p_ens * r)
    print(f"  {ra['name']} + {rb['name']}: sharpe={sh:.3f}  CI=[{lo:.3f},{hi:.3f}]  "
          f"p_std={p_ens.std():.3f}")


if __name__ == "__main__":
    main()
