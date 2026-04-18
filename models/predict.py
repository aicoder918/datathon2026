"""End-to-end: inner CV on train to pick {alpha, gamma, lambda}, refit on all
train, predict positions for public + private test, write submission CSV.

Usage:
    python -m models.predict                 # pick (alpha, gamma, lam) via inner CV
    python -m models.predict --lam 0.5       # override lambda (keep alpha/gamma from CV)
    python -m models.predict --constant      # emit constant-long baseline only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.features import (
    DATA_DIR, build_features, compute_train_target, load_finbert,
)
from models.sharpe_model import (
    MeanHead, VarianceHead, positions_from_heads, sharpe, STABLE_FEATURES,
)
from models.cv_evaluate import (
    ALPHAS, GAMMA_GRID, LAMBDAS, _fit_predict_fold,
)

ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS = ROOT / "submissions"
INNER_SPLITS = 5


def pick_hyperparams(X: pd.DataFrame, r: pd.Series, features: list = None,
                     alpha_grid: list = None):
    """Inner 5-fold CV on all training sessions, selecting (alpha, gamma, lam)
    jointly by pooled OOF Sharpe. Uses given feature subset for the mean head."""
    if features is None:
        features = list(STABLE_FEATURES)
    if alpha_grid is None:
        alpha_grid = [10.0, 30.0, 100.0, 300.0]
    sessions = X.index.to_numpy()
    r_arr = r.reindex(X.index).to_numpy()
    gkf = GroupKFold(n_splits=INNER_SPLITS)
    best = {"alpha": alpha_grid[len(alpha_grid)//2], "gamma": 0.0, "lam": 0.0,
            "sharpe": -1e9}
    for alpha in alpha_grid:
        mu_oof = np.zeros(len(X)); s2_oof = np.zeros(len(X))
        ref_parts = []
        for tr_idx, va_idx in gkf.split(X, groups=sessions):
            X_t = X.iloc[tr_idx]; r_t = r_arr[tr_idx]
            X_v = X.iloc[va_idx]
            mh = MeanHead(alpha=alpha, features=list(features)).fit(X_t, r_t)
            vh = VarianceHead(alpha=5.0).fit(X_t, r_t)
            mu_va = mh.predict(X_v); s2_va = vh.predict(X_v)
            mu_tr = mh.predict(X_t); s2_tr = vh.predict(X_t)
            mu_oof[va_idx] = mu_va; s2_oof[va_idx] = s2_va
            ref_parts.append((mu_tr + 0.0) / np.maximum(s2_tr, 1e-12))
        ref_raw = np.concatenate(ref_parts)
        for gamma in GAMMA_GRID:
            for lam in LAMBDAS:
                p = positions_from_heads(mu_oof, s2_oof, gamma=gamma, lam=lam,
                                         ref_raw=ref_raw)
                sh = sharpe(p * r_arr)
                if sh > best["sharpe"]:
                    best = {"alpha": alpha, "gamma": gamma, "lam": lam, "sharpe": sh}
    return best


def build_submission_df(test_sessions: np.ndarray, positions: np.ndarray) -> pd.DataFrame:
    assert len(test_sessions) == len(positions)
    return pd.DataFrame({"session": test_sessions.astype(int),
                          "target_position": positions.astype(float)}) \
        .sort_values("session").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=None,
                    help="Override lambda (else picked by inner CV).")
    ap.add_argument("--gamma", type=float, default=None,
                    help="Override gamma (else picked by inner CV).")
    ap.add_argument("--constant", action="store_true",
                    help="Emit constant-long submission (p=1) and skip model.")
    ap.add_argument("--out", type=str, default="sharpe_v1.csv")
    args = ap.parse_args()

    fb = load_finbert()
    bs = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    bu = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    hs = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
    X_tr = build_features(bs, hs, finbert=fb)
    r = compute_train_target(bs, bu).reindex(X_tr.index)

    pub_bars = pd.read_parquet(DATA_DIR / "bars_seen_public_test.parquet")
    pub_head = pd.read_parquet(DATA_DIR / "headlines_seen_public_test.parquet")
    prv_bars = pd.read_parquet(DATA_DIR / "bars_seen_private_test.parquet")
    prv_head = pd.read_parquet(DATA_DIR / "headlines_seen_private_test.parquet")
    X_pub = build_features(pub_bars, pub_head, finbert=fb)
    X_prv = build_features(prv_bars, prv_head, finbert=fb)

    all_test_sessions = np.concatenate([X_pub.index.to_numpy(), X_prv.index.to_numpy()])
    print(f"train: {len(X_tr)}  public_test: {len(X_pub)}  private_test: {len(X_prv)}")

    SUBMISSIONS.mkdir(exist_ok=True)
    if args.constant:
        positions = np.ones(len(all_test_sessions))
        out = build_submission_df(all_test_sessions, positions)
        path = SUBMISSIONS / args.out
        out.to_csv(path, index=False)
        print(f"Wrote {path} (constant-long baseline, n={len(out)})")
        return

    features = list(STABLE_FEATURES)
    print(f"mean-head features ({len(features)}): {features}")
    chosen = pick_hyperparams(X_tr, r, features=features)
    if args.lam is not None:
        chosen["lam"] = args.lam
    if args.gamma is not None:
        chosen["gamma"] = args.gamma
    print(f"Selected: alpha={chosen['alpha']}  gamma={chosen['gamma']:.4f}  "
          f"lam={chosen['lam']}  inner_sharpe={chosen['sharpe']:.3f}")

    # Refit on all 1000 training sessions
    r_arr = r.to_numpy()
    mh = MeanHead(alpha=chosen["alpha"], features=features).fit(X_tr, r_arr)
    vh = VarianceHead(alpha=5.0).fit(X_tr, r_arr)
    mu_tr = mh.predict(X_tr); s2_tr = vh.predict(X_tr)
    raw_tr = (mu_tr + chosen["gamma"]) / np.maximum(s2_tr, 1e-12)

    # Predict test
    X_test = pd.concat([X_pub, X_prv])
    mu_te = mh.predict(X_test); s2_te = vh.predict(X_test)
    positions = positions_from_heads(
        mu_te, s2_te,
        gamma=chosen["gamma"], lam=chosen["lam"],
        ref_raw=raw_tr,
    )

    out = build_submission_df(X_test.index.to_numpy(), positions)
    path = SUBMISSIONS / args.out
    out.to_csv(path, index=False)
    print(f"Wrote {path} (n={len(out)})")
    print(f"Position stats: mean={out['target_position'].mean():.3f}  "
          f"std={out['target_position'].std():.3f}  "
          f"min={out['target_position'].min():.3f}  "
          f"max={out['target_position'].max():.3f}  "
          f"median={out['target_position'].median():.3f}")


if __name__ == "__main__":
    main()
