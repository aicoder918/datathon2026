"""Test structurally different position strategies (not just feature tweaks).

Goal: find a submission MEANINGFULLY different from v1 (not just a CV microvariant).
v1 is Ridge α=100, stable13, λ=1, γ=0.006.

Variants:
  - v4_highreg:  Ridge α=300 (more conservative positions)
  - v4_allreg:   Ridge α=300 on ALL 22 features (use regularization for selection)
  - v4_voltarget: pure 1/σ² (mean head off)
  - v4_catridge_ens: Ridge + CatBoost position average
  - v4_lowlam:   Ridge α=100 stable13 with λ=0.75 (more shrinkage to constant)
  - v4_lamsweep: Ridge α=100 stable13 sweep λ ∈ {0.25, 0.5, 0.75, 1.0}
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
    MeanHead, CatMeanHead, VarianceHead, positions_from_heads, sharpe, STABLE_FEATURES,
)
from models.cv_evaluate import bootstrap_sharpe

OUTER = 10
GAMMA = 0.006


def _cv_predict(X: pd.DataFrame, r: np.ndarray, sessions: np.ndarray,
                factory, feature_cols: list) -> dict:
    gkf = GroupKFold(n_splits=OUTER)
    mu_oof = np.zeros(len(X))
    s2_oof = np.zeros(len(X))
    ref_parts = []
    for tr_idx, va_idx in gkf.split(X, groups=sessions):
        X_tr = X.iloc[tr_idx]; r_tr = r[tr_idx]
        X_va = X.iloc[va_idx]
        mh = factory().fit(X_tr, r_tr)
        vh = VarianceHead(alpha=5.0).fit(X_tr, r_tr)
        mu_va = mh.predict(X_va); s2_va = vh.predict(X_va)
        mu_tr = mh.predict(X_tr); s2_tr = vh.predict(X_tr)
        mu_oof[va_idx] = mu_va; s2_oof[va_idx] = s2_va
        ref_parts.append((mu_tr + GAMMA) / np.maximum(s2_tr, 1e-12))
    return {"mu_oof": mu_oof, "s2_oof": s2_oof,
            "ref_raw": np.concatenate(ref_parts)}


def _score(name: str, r: np.ndarray, positions: np.ndarray,
           bs_sh: float, bs_se: float):
    sh, lo, hi, se = bootstrap_sharpe(positions * r)
    print(f"  {name:<24}: sharpe={sh:.3f}  CI=[{lo:.3f},{hi:.3f}]  "
          f"delta={sh-bs_sh:+.3f} ({(sh-bs_sh)/bs_se:+.2f} SE)  "
          f"p=[{positions.min():.2f},{positions.max():.2f}]  "
          f"p_std={positions.std():.3f}")
    return sh


def main():
    bs = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    bu = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    hs = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
    fb = load_finbert()

    X = build_features(bs, hs, finbert=fb)
    X = X.drop(columns=["tpl_late_score", "tpl_all_score"])
    r_ser = compute_train_target(bs, bu).reindex(X.index)
    r = r_ser.to_numpy()
    sessions = X.index.to_numpy()

    bs_sh, _, _, bs_se = bootstrap_sharpe(1.0 * r)
    print(f"constant_long: sharpe={bs_sh:.3f}  SE={bs_se:.3f}\n")

    stable = list(STABLE_FEATURES)
    all22 = list(PRICE_FEATURES) + list(HEADLINE_FEATURES)

    # Run CV for each candidate mean head
    print("Running CVs (one call per model)...")
    ridge_stable_a100 = _cv_predict(X, r, sessions,
                                    lambda: MeanHead(alpha=100.0, features=stable), stable)
    ridge_stable_a300 = _cv_predict(X, r, sessions,
                                    lambda: MeanHead(alpha=300.0, features=stable), stable)
    ridge_all_a300 = _cv_predict(X, r, sessions,
                                 lambda: MeanHead(alpha=300.0, features=all22), all22)
    ridge_all_a1000 = _cv_predict(X, r, sessions,
                                  lambda: MeanHead(alpha=1000.0, features=all22), all22)
    cat_stable = _cv_predict(X, r, sessions,
                             lambda: CatMeanHead(depth=3, iterations=300,
                                                 l2_leaf_reg=10.0, features=stable), stable)
    cat_stable_d2 = _cv_predict(X, r, sessions,
                                lambda: CatMeanHead(depth=2, iterations=400,
                                                    l2_leaf_reg=20.0, features=stable), stable)

    def p(out, gamma=GAMMA, lam=1.0):
        return positions_from_heads(out["mu_oof"], out["s2_oof"], gamma=gamma, lam=lam,
                                    ref_raw=out["ref_raw"])

    print("\n=== Single-model candidates ===")
    _score("v1_ridge_s13_a100_lam1", r, p(ridge_stable_a100), bs_sh, bs_se)
    _score("ridge_s13_a100_lam0.75", r, p(ridge_stable_a100, lam=0.75), bs_sh, bs_se)
    _score("ridge_s13_a100_lam0.5", r, p(ridge_stable_a100, lam=0.5), bs_sh, bs_se)
    _score("ridge_s13_a300_lam1", r, p(ridge_stable_a300), bs_sh, bs_se)
    _score("ridge_all_a300_lam1", r, p(ridge_all_a300), bs_sh, bs_se)
    _score("ridge_all_a1000_lam1", r, p(ridge_all_a1000), bs_sh, bs_se)
    _score("cat_s13_d3_lam1", r, p(cat_stable), bs_sh, bs_se)
    _score("cat_s13_d2_lam1", r, p(cat_stable_d2), bs_sh, bs_se)

    # Vol-targeted only (no mean head)
    s2_oof_vt = ridge_stable_a100["s2_oof"]
    p_vt = 1.0 / np.maximum(s2_oof_vt, 1e-12)
    # Normalize to mean = 1 so it's comparable to λ=1 constant-long
    p_vt = p_vt / np.median(p_vt)
    _score("vol_target_only", r, p_vt, bs_sh, bs_se)

    print("\n=== Ensembles (equal-weight average of positions) ===")
    p_r100 = p(ridge_stable_a100)
    p_r300 = p(ridge_stable_a300)
    p_rall = p(ridge_all_a300)
    p_cat = p(cat_stable)
    p_cat2 = p(cat_stable_d2)

    _score("ENS: r100+r300 (s13)", r, 0.5*p_r100 + 0.5*p_r300, bs_sh, bs_se)
    _score("ENS: r100+cat_s13", r, 0.5*p_r100 + 0.5*p_cat, bs_sh, bs_se)
    _score("ENS: r100+cat_d2", r, 0.5*p_r100 + 0.5*p_cat2, bs_sh, bs_se)
    _score("ENS: r100+r_all_a300+cat", r, (p_r100 + p_rall + p_cat) / 3.0, bs_sh, bs_se)
    _score("ENS: r100+const(1.0)", r, 0.5*p_r100 + 0.5*1.0, bs_sh, bs_se)
    _score("ENS: r100+vt", r, 0.5*p_r100 + 0.5*p_vt, bs_sh, bs_se)


if __name__ == "__main__":
    main()
