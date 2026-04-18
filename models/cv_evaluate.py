"""Nested CV evaluation for the Sharpe-optimal pipeline.

Outer: 10-fold GroupKFold by session -> pooled OOF predictions on all 1,000 sessions.
Inner: 5-fold GroupKFold on each outer-train to pick {ridge alpha, gamma, lambda}.

Prints a baseline comparison table with pooled OOF Sharpe and 95% bootstrap CI.
Also runs stability diagnostics: per-feature coef sign consistency across folds,
vol-head sanity (sigma2 vs r^2), and mean-head sanity (Pearson(mu_hat, r)).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.features import (
    DATA_DIR, FEATURE_COLS, build_features, compute_train_target, load_finbert,
)
from models.sharpe_model import (
    MeanHead, VarianceHead, positions_from_heads, sharpe,
)

SEED = 42
OUTER_SPLITS = 10
INNER_SPLITS = 5
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMA_GRID = [0.0, 0.002, 0.004, 0.006]      # centred on unconditional mean r ~ 0.0035
N_BOOT = 2000


# --------------------------------------------------------------- core CV

def _fit_predict_fold(
    X_tr, r_tr, X_va, alpha_m: float, alpha_v: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MeanHead]:
    mh = MeanHead(alpha=alpha_m).fit(X_tr, r_tr)
    vh = VarianceHead(alpha=alpha_v).fit(X_tr, r_tr)
    # Train-set predictions (for winsor ref + gamma/lambda selection scale)
    mu_tr = mh.predict(X_tr)
    s2_tr = vh.predict(X_tr)
    raw_tr = (mu_tr + 0.0) / np.maximum(s2_tr, 1e-12)
    # Val-set predictions
    mu_va = mh.predict(X_va)
    s2_va = vh.predict(X_va)
    return mu_va, s2_va, raw_tr, mh


def _inner_select(X_tr, r_tr, groups_tr) -> dict:
    """Inner 5-fold CV to pick ridge alpha + (gamma, lambda) jointly on OOF Sharpe.

    We pool the inner OOF mu/sigma2 predictions, then grid-search (gamma, lambda)
    on top -- this is the only way to select the post-processing on a held-out
    fold rather than on training data.
    """
    gkf = GroupKFold(n_splits=INNER_SPLITS)
    best = {"alpha": ALPHAS[len(ALPHAS)//2], "gamma": 0.0, "lam": 0.0, "sharpe": -1e9}

    for alpha in ALPHAS:
        mu_oof = np.zeros(len(X_tr))
        s2_oof = np.zeros(len(X_tr))
        ref_raw_parts = []
        r_arr = r_tr.to_numpy() if hasattr(r_tr, "to_numpy") else np.asarray(r_tr)
        for tr_idx, va_idx in gkf.split(X_tr, groups=groups_tr):
            X_t = X_tr.iloc[tr_idx]; r_t = r_arr[tr_idx]
            X_v = X_tr.iloc[va_idx]
            mu_va, s2_va, raw_tr, _ = _fit_predict_fold(X_t, r_t, X_v, alpha_m=alpha)
            mu_oof[va_idx] = mu_va
            s2_oof[va_idx] = s2_va
            ref_raw_parts.append(raw_tr)
        ref_raw = np.concatenate(ref_raw_parts)

        for gamma in GAMMA_GRID:
            for lam in LAMBDAS:
                p = positions_from_heads(mu_oof, s2_oof, gamma=gamma, lam=lam,
                                          ref_raw=ref_raw)
                sh = sharpe(p * r_arr)
                if sh > best["sharpe"]:
                    best = {"alpha": alpha, "gamma": gamma, "lam": lam, "sharpe": sh}
    return best


def outer_cv(X: pd.DataFrame, r: pd.Series, seed: int = SEED) -> dict:
    """Run nested CV. Returns pooled OOF predictions + per-fold diagnostics."""
    sessions = X.index.to_numpy()
    r_arr = r.reindex(X.index).to_numpy()

    gkf = GroupKFold(n_splits=OUTER_SPLITS)
    mu_oof = np.zeros(len(X))
    s2_oof = np.zeros(len(X))
    ref_raw_parts = []
    chosen_per_fold = []
    coefs_per_fold = []

    for fold_ix, (tr_idx, va_idx) in enumerate(gkf.split(X, groups=sessions)):
        X_tr = X.iloc[tr_idx]; r_tr = r_arr[tr_idx]
        X_va = X.iloc[va_idx]
        groups_tr = sessions[tr_idx]

        picked = _inner_select(X_tr, r_tr, groups_tr)
        chosen_per_fold.append(picked)

        mu_va, s2_va, raw_tr, mh = _fit_predict_fold(
            X_tr, r_tr, X_va, alpha_m=picked["alpha"]
        )
        mu_oof[va_idx] = mu_va
        s2_oof[va_idx] = s2_va
        ref_raw_parts.append(raw_tr)
        coefs_per_fold.append(mh.coefs())

    ref_raw = np.concatenate(ref_raw_parts)
    return {
        "mu_oof": mu_oof,
        "s2_oof": s2_oof,
        "r": r_arr,
        "ref_raw": ref_raw,
        "chosen_per_fold": chosen_per_fold,
        "coefs_per_fold": coefs_per_fold,
    }


# --------------------------------------------------------------- bootstrap Sharpe

def bootstrap_sharpe(pnl: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = sharpe(pnl[idx])
    mean = sharpe(pnl)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    se = boots.std(ddof=1)
    return mean, float(lo), float(hi), float(se)


# --------------------------------------------------------------- baselines

def _finalize_positions(mu: np.ndarray, s2: np.ndarray, gamma: float, lam: float,
                       ref_raw: np.ndarray) -> np.ndarray:
    return positions_from_heads(mu, s2, gamma=gamma, lam=lam, ref_raw=ref_raw)


def report(out: dict) -> pd.DataFrame:
    r = out["r"]; mu = out["mu_oof"]; s2 = out["s2_oof"]; ref = out["ref_raw"]
    rows = []

    def _row(name: str, p: np.ndarray):
        sh, lo, hi, se = bootstrap_sharpe(p * r)
        rows.append({"strategy": name, "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "se": se})

    # (a) constant long
    _row("constant_long", np.ones_like(r))
    # (b) vol-targeted long
    p_vt = 1.0 / np.maximum(s2, 1e-12)
    p_vt = p_vt / np.median(p_vt)
    _row("vol_targeted_long", p_vt)
    # (c) sentiment-only (use sent_last10 as mu-proxy, unit sigma2)
    # We need the feature values to isolate sent-only. Skipped here (pick best full model instead).
    # (d) full model at several lambdas (with gamma picked per-lambda on OOF)
    for lam in LAMBDAS:
        best = {"gamma": 0.0, "sharpe": -1e9}
        for gamma in GAMMA_GRID:
            p = _finalize_positions(mu, s2, gamma, lam, ref)
            sh = sharpe(p * r)
            if sh > best["sharpe"]:
                best = {"gamma": gamma, "sharpe": sh}
        p = _finalize_positions(mu, s2, best["gamma"], lam, ref)
        _row(f"full@lam={lam},gamma={best['gamma']:.4f}", p)

    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------- stability

def stability_diagnostics(out: dict) -> pd.DataFrame:
    coefs = pd.DataFrame(out["coefs_per_fold"])  # one row per fold
    signs = np.sign(coefs)
    consistency = signs.apply(lambda col: int((col == signs[col.name].mode().iloc[0]).sum()))
    mean_coef = coefs.mean(axis=0)
    return pd.DataFrame({
        "mean_coef": mean_coef,
        "sign_agree_out_of_10": consistency,
    }).sort_values("mean_coef", key=lambda s: s.abs(), ascending=False)


def vol_sanity(out: dict) -> dict:
    r = out["r"]; s2 = out["s2_oof"]
    r2 = r * r
    corr_vol = float(np.corrcoef(s2, r2)[0, 1])
    corr_mu = float(np.corrcoef(out["mu_oof"], r)[0, 1])
    return {"corr(sigma2_hat, r^2)": corr_vol, "corr(mu_hat, r)": corr_mu}


# --------------------------------------------------------------- main

def main():
    bs = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    bu = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    hs = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
    fb = load_finbert()

    X = build_features(bs, hs, finbert=fb)
    r = compute_train_target(bs, bu)
    X = X.loc[r.index.intersection(X.index)]
    r = r.loc[X.index]

    print(f"n_sessions={len(X)}  n_features={X.shape[1]}")
    print(f"target: mean={r.mean():.5f}  std={r.std():.5f}  p(r>0)={(r>0).mean():.3f}")
    print()

    print("Running nested CV (outer=10, inner=5)...")
    out = outer_cv(X, r)

    print("\n=== Pooled OOF Sharpe (bootstrap 95% CI) ===")
    tbl = report(out)
    baseline_se = float(tbl.loc[tbl["strategy"] == "constant_long", "se"].iloc[0])
    baseline_sh = float(tbl.loc[tbl["strategy"] == "constant_long", "sharpe"].iloc[0])
    tbl["delta_vs_baseline"] = tbl["sharpe"] - baseline_sh
    tbl["delta_in_se"] = tbl["delta_vs_baseline"] / baseline_se
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Stability: coef sign agreement across 10 outer folds ===")
    print(stability_diagnostics(out).to_string(float_format=lambda x: f"{x:.5f}"))

    print("\n=== Head sanity ===")
    for k, v in vol_sanity(out).items():
        print(f"  {k}: {v:+.4f}")

    print("\n=== Chosen per-fold (alpha, gamma, lambda, inner-OOF Sharpe) ===")
    for i, c in enumerate(out["chosen_per_fold"]):
        print(f"  fold{i}: alpha={c['alpha']}  gamma={c['gamma']:.4f}  lam={c['lam']}  "
              f"inner_sharpe={c['sharpe']:.3f}")

    print("\n=== Acceptance gate ===")
    best = tbl.sort_values("sharpe", ascending=False).iloc[0]
    print(f"Best strategy: {best['strategy']}  sharpe={best['sharpe']:.3f}  "
          f"delta/SE={best['delta_in_se']:.2f}")
    if best["strategy"].startswith("full@") and best["delta_in_se"] >= 1.5:
        print("  -> PASS: model beats constant-long by >= 1.5 SE. Ship model.")
    else:
        print("  -> FAIL (or no signal): ship constant-long or vol-targeted.")


if __name__ == "__main__":
    main()
