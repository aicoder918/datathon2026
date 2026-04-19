"""Template + direction + vol baseline, weights fit on a proxy task across
all 21k sessions (train + public + private).

Proxy task (computable on every split — only needs bars 0..49):
    features on bars  0..24       target = close[49]/close[24] - 1
Real task (only on train):
    features on bars  0..49       target = close[99]/close[49] - 1

Hypothesis: the linear relationship between (template, direction, vol) and the
forward return is stationary across horizons, so coefficients fit on the proxy
transfer to the real task. The proxy gives us 21× more data for the fit.

At test time we z-score full-window features using the combined cross-split
full-window stats, apply the proxy-fit coefficients, then tilt around 1.0 with
a train-tuned alpha.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model.data_seq import (
    compute_template_impacts,
    load_bars,
    load_headlines,
)


# --------------------------------------------------------------------------- #
# Per-session feature builders
# --------------------------------------------------------------------------- #

def session_template_score_range(hls_df: pd.DataFrame,
                                 template_impacts: np.ndarray,
                                 lo: int, hi: int) -> dict:
    """Sum of template impacts for headlines firing in bars [lo, hi)."""
    n = len(template_impacts)
    out = {}
    for s, g in hls_df.groupby("session"):
        mask = (g["bar_ix"] >= lo) & (g["bar_ix"] < hi)
        tids = g.loc[mask, "template_index"].to_numpy(dtype=int)
        tids = tids[(tids >= 0) & (tids < n)]
        out[int(s)] = float(template_impacts[tids].sum()) if len(tids) else 0.0
    return out


def session_direction_range(bars_df: pd.DataFrame,
                            lo: int, hi: int) -> dict:
    """close[hi-1] / close[lo] - 1."""
    out = {}
    for s, g in bars_df.groupby("session"):
        g = g.sort_values("bar_ix")
        c = g["close"].to_numpy(dtype=float)
        if len(c) <= hi - 1 or len(c) <= lo:
            out[int(s)] = 0.0
            continue
        out[int(s)] = float(c[hi - 1] / (c[lo] + 1e-8) - 1.0)
    return out


def session_vol_range(bars_df: pd.DataFrame,
                      lo: int, hi: int) -> dict:
    """std of 1-bar returns over bars [lo, hi)."""
    out = {}
    for s, g in bars_df.groupby("session"):
        g = g.sort_values("bar_ix").iloc[lo:hi]
        c = g["close"].to_numpy(dtype=float)
        if len(c) < 2:
            out[int(s)] = 0.0
            continue
        ret = c[1:] / (c[:-1] + 1e-8) - 1.0
        out[int(s)] = float(np.std(ret))
    return out


def session_forward_return(bars_df: pd.DataFrame,
                           lo: int, hi: int) -> dict:
    """close[hi-1] / close[lo] - 1. Same form as direction, but used as target."""
    return session_direction_range(bars_df, lo, hi)


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #

def sharpe(pos, ret) -> float:
    pos = np.asarray(pos, dtype=float)
    ret = np.asarray(ret, dtype=float)
    pnl = pos * ret
    std = pnl.std()
    return 0.0 if std < 1e-12 else float(pnl.mean() / std * 16)


def zstats(a: np.ndarray) -> tuple:
    return float(a.mean()), float(a.std())


def zscore(a: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (a - mu) / (sigma + 1e-12)


# --------------------------------------------------------------------------- #
# Proxy-task OLS
# --------------------------------------------------------------------------- #

def build_feature_tuple(template_impacts, lo, hi, bars_dict, hls_dict):
    """For each split, return dict: session -> (tpl, dir, vol).
    lo, hi defines the feature window bars [lo, hi)."""
    out = {}
    for split, bars in bars_dict.items():
        hls = hls_dict[split]
        tpl = session_template_score_range(hls, template_impacts, lo, hi)
        dir_ = session_direction_range(bars, lo, hi)
        vol = session_vol_range(bars, lo, hi)
        sessions = sorted(int(s) for s in bars["session"].unique())
        feats = {int(s): (tpl.get(int(s), 0.0),
                          dir_.get(int(s), 0.0),
                          vol.get(int(s), 0.0))
                 for s in sessions}
        out[split] = (sessions, feats)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="submission_proxy_fit.csv")
    parser.add_argument("--first-hi", type=int, default=25,
                        help="Proxy feature window upper bound (exclusive).")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha (default: pick from train sweep).")
    args = parser.parse_args()

    FIRST_LO, FIRST_HI = 0, args.first_hi        # proxy features: bars [0, FIRST_HI)
    PROXY_LO, PROXY_HI = args.first_hi, 50       # proxy target:   close[49]/close[FIRST_HI]-1
    FULL_LO, FULL_HI = 0, 50                     # real features:  bars [0, 50)
    REAL_LO, REAL_HI = 50, 100                   # real target:    close[99]/close[49]-1

    print("Computing template impacts (train-only)...")
    bars_full_train = load_bars("train_full")
    hls_train = load_headlines("train")
    template_impacts, counts = compute_template_impacts(bars_full_train, hls_train)
    print(f"  impacts: min={template_impacts.min():+.5f} "
          f"max={template_impacts.max():+.5f} "
          f"mean={template_impacts.mean():+.5f} "
          f"(non-zero templates: {int((counts > 0).sum())})")

    print("\nLoading bars + headlines for all splits...")
    bars_dict = {
        "train": bars_full_train,
        "public": load_bars("public"),
        "private": load_bars("private"),
    }
    hls_dict = {
        "train": hls_train,
        "public": load_headlines("public"),
        "private": load_headlines("private"),
    }
    for k, df in bars_dict.items():
        print(f"  {k}: {df['session'].nunique()} sessions, {len(df)} bars")

    # ------------------------------------------------------------------ #
    # Proxy features on bars [FIRST_LO, FIRST_HI) across all splits.
    # Proxy target = close[PROXY_HI-1] / close[PROXY_LO] - 1.
    # ------------------------------------------------------------------ #
    print(f"\n=== Proxy fit (features bars {FIRST_LO}..{FIRST_HI-1}, "
          f"target close[{PROXY_HI-1}]/close[{PROXY_LO}]-1) ===")
    proxy_feats = build_feature_tuple(template_impacts, FIRST_LO, FIRST_HI,
                                      bars_dict, hls_dict)
    full_feats = build_feature_tuple(template_impacts, FULL_LO, FULL_HI,
                                     bars_dict, hls_dict)
    # Proxy target (available for every split since close[49] is visible everywhere).
    proxy_target_dict = {}
    for split, bars in bars_dict.items():
        proxy_target_dict[split] = session_forward_return(bars, PROXY_LO, PROXY_HI)

    # Stack proxy features + targets across all splits.
    all_tpl, all_dir, all_vol, all_y = [], [], [], []
    per_split_counts = {}
    for split in ("train", "public", "private"):
        sessions, feats = proxy_feats[split]
        ys = proxy_target_dict[split]
        per_split_counts[split] = len(sessions)
        for s in sessions:
            t, d, v = feats[s]
            all_tpl.append(t)
            all_dir.append(d)
            all_vol.append(v)
            all_y.append(ys.get(s, 0.0))
    tpl = np.array(all_tpl)
    dir_ = np.array(all_dir)
    vol = np.array(all_vol)
    y_p = np.array(all_y)
    print(f"  total proxy sessions: {len(y_p)} "
          f"({per_split_counts})")
    print(f"  proxy target: mean={y_p.mean():+.5f}, std={y_p.std():.5f}")

    # Z-score proxy features using the combined cross-split stats.
    tpl_mu_p, tpl_sd_p = zstats(tpl)
    dir_mu_p, dir_sd_p = zstats(dir_)
    vol_mu_p, vol_sd_p = zstats(vol)
    z_tpl_p = zscore(tpl, tpl_mu_p, tpl_sd_p)
    z_dir_p = zscore(dir_, dir_mu_p, dir_sd_p)
    z_vol_p = zscore(vol, vol_mu_p, vol_sd_p)
    print(f"  proxy template: mu={tpl_mu_p:+.6f}, sd={tpl_sd_p:.6f}")
    print(f"  proxy dir:      mu={dir_mu_p:+.6f}, sd={dir_sd_p:.6f}")
    print(f"  proxy vol:      mu={vol_mu_p:+.6f}, sd={vol_sd_p:.6f}")
    print(f"  corr(tpl_p, y_p) = {np.corrcoef(tpl, y_p)[0,1]:+.4f}")
    print(f"  corr(dir_p, y_p) = {np.corrcoef(dir_, y_p)[0,1]:+.4f}")
    print(f"  corr(vol_p, y_p) = {np.corrcoef(vol, y_p)[0,1]:+.4f}")

    # OLS fit (no intercept — features are z-scored).
    X = np.stack([z_tpl_p, z_dir_p, z_vol_p], axis=1)  # (N, 3)
    w, *_ = np.linalg.lstsq(X, y_p, rcond=None)
    w_tpl, w_dir, w_vol = [float(v) for v in w]
    y_hat_p = X @ w
    r2 = 1.0 - np.sum((y_p - y_hat_p) ** 2) / np.sum((y_p - y_p.mean()) ** 2)
    print(f"\n  OLS weights (z-scored features, no intercept):")
    print(f"    w_tpl = {w_tpl:+.6f}")
    print(f"    w_dir = {w_dir:+.6f}")
    print(f"    w_vol = {w_vol:+.6f}")
    print(f"    R^2   = {r2:+.4f}")
    beta = w_dir / w_tpl if abs(w_tpl) > 1e-12 else 0.0
    gamma = w_vol / w_tpl if abs(w_tpl) > 1e-12 else 0.0
    print(f"  Derived (relative to w_tpl): beta={beta:+.4f}, gamma={gamma:+.4f}")

    # ------------------------------------------------------------------ #
    # Real features on bars [0, 50) across all splits.
    # Real target = close[99]/close[49] - 1, available only on train.
    # ------------------------------------------------------------------ #
    print(f"\n=== Applying proxy-fit weights to real features (bars 0..49) ===")
    all_tpl_r, all_dir_r, all_vol_r = [], [], []
    split_slices = {}
    cursor = 0
    for split in ("train", "public", "private"):
        sessions, feats = full_feats[split]
        for s in sessions:
            t, d, v = feats[s]
            all_tpl_r.append(t)
            all_dir_r.append(d)
            all_vol_r.append(v)
        split_slices[split] = (cursor, cursor + len(sessions), sessions)
        cursor += len(sessions)
    tpl_r = np.array(all_tpl_r)
    dir_r = np.array(all_dir_r)
    vol_r = np.array(all_vol_r)

    tpl_mu_r, tpl_sd_r = zstats(tpl_r)
    dir_mu_r, dir_sd_r = zstats(dir_r)
    vol_mu_r, vol_sd_r = zstats(vol_r)
    z_tpl_r = zscore(tpl_r, tpl_mu_r, tpl_sd_r)
    z_dir_r = zscore(dir_r, dir_mu_r, dir_sd_r)
    z_vol_r = zscore(vol_r, vol_mu_r, vol_sd_r)
    print(f"  real template: mu={tpl_mu_r:+.6f}, sd={tpl_sd_r:.6f}")
    print(f"  real dir:      mu={dir_mu_r:+.6f}, sd={dir_sd_r:.6f}")
    print(f"  real vol:      mu={vol_mu_r:+.6f}, sd={vol_sd_r:.6f}")

    y_hat_r = w_tpl * z_tpl_r + w_dir * z_dir_r + w_vol * z_vol_r

    # Train-only: evaluate real-target correlations + tune alpha.
    tr_lo, tr_hi, tr_sessions = split_slices["train"]
    real_target_map = session_forward_return(bars_full_train, REAL_LO, REAL_HI)
    # trim train sessions to those with 100 bars (i.e., target available)
    train_valid_mask = np.array([s in real_target_map for s in tr_sessions])
    tr_idx = np.where(train_valid_mask)[0]
    y_r = np.array([real_target_map[tr_sessions[i]] for i in tr_idx])
    z_tpl_tr = z_tpl_r[tr_lo:tr_hi][tr_idx]
    z_dir_tr = z_dir_r[tr_lo:tr_hi][tr_idx]
    z_vol_tr = z_vol_r[tr_lo:tr_hi][tr_idx]
    y_hat_tr = y_hat_r[tr_lo:tr_hi][tr_idx]

    print(f"\n  Train sessions with real target: {len(tr_idx)}")
    print(f"  real target: mean={y_r.mean():+.5f}, std={y_r.std():.5f}")
    print(f"  corr(tpl_r, y_r) = {np.corrcoef(z_tpl_tr, y_r)[0,1]:+.4f}")
    print(f"  corr(dir_r, y_r) = {np.corrcoef(z_dir_tr, y_r)[0,1]:+.4f}")
    print(f"  corr(vol_r, y_r) = {np.corrcoef(z_vol_tr, y_r)[0,1]:+.4f}")
    print(f"  corr(proxy_yhat, y_r) = {np.corrcoef(y_hat_tr, y_r)[0,1]:+.4f}")

    # Tilt y_hat around 1.0 and sweep alpha on train real Sharpe.
    # Express as:  position = 1 + alpha * (z_tpl + beta*z_dir + gamma*z_vol)
    # Equivalently: position = 1 + alpha' * y_hat_r, with alpha' = alpha / w_tpl.
    # We'll sweep both the combined (alpha) and the proxy-fit (alpha') forms.
    print("\n  ─ alpha sweep (train real target) ─")
    print(f"    constant 1.0   : {sharpe(np.ones_like(y_r), y_r):.3f}")
    best = (-np.inf, None, None)
    # Direct scaling of the OLS prediction.
    for alpha_p in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        pos = 1.0 + alpha_p * y_hat_tr
        s = sharpe(pos, y_r)
        print(f"    1 + {alpha_p:>5.1f} * yhat   : {s:.3f}")
        if s > best[0]:
            best = (s, alpha_p, "direct")
    # Normalized form.
    for a in [0.1, 0.25, 0.5, 1.0]:
        combined = z_tpl_tr + beta * z_dir_tr + gamma * z_vol_tr
        pos = 1.0 + a * combined
        s = sharpe(pos, y_r)
        print(f"    1 + {a:>5.2f} * (tpl + {beta:+.2f}*dir + {gamma:+.2f}*vol): {s:.3f}")
        if s > best[0]:
            best = (s, a, "normalized")
    print(f"  best: {best}")

    # ------------------------------------------------------------------ #
    # Build submission.
    # ------------------------------------------------------------------ #
    alpha = args.alpha if args.alpha is not None else (
        best[1] if best[2] == "direct" else best[1]
    )
    mode = best[2]
    print(f"\nBuilding submission (mode={mode}, alpha={alpha}, "
          f"beta={beta:+.4f}, gamma={gamma:+.4f})")

    parts = []
    for split in ("public", "private"):
        lo_, hi_, sessions = split_slices[split]
        ytr_s = y_hat_r[lo_:hi_]
        z_tpl_s = z_tpl_r[lo_:hi_]
        z_dir_s = z_dir_r[lo_:hi_]
        z_vol_s = z_vol_r[lo_:hi_]
        if mode == "direct":
            pos = 1.0 + alpha * ytr_s
        else:
            pos = 1.0 + alpha * (z_tpl_s + beta * z_dir_s + gamma * z_vol_s)
        print(f"  {split}: n={len(sessions)}, "
              f"pos mean={pos.mean():.3f}, std={pos.std():.3f}, "
              f"min={pos.min():.3f}, max={pos.max():.3f}, "
              f"frac neg={float((pos < 0).mean()):.1%}")
        parts.append(pd.DataFrame({"session": np.array(sessions, dtype=int),
                                   "target_position": pos}))

    sub = (pd.concat(parts, ignore_index=True)
             .sort_values("session")
             .reset_index(drop=True))
    assert sub["session"].is_unique, "duplicate sessions"
    assert sub["target_position"].notna().all(), "NaN in positions"
    sub.to_csv(args.out, index=False)
    print(f"\nSaved submission ({len(sub)} rows) -> {args.out}")


if __name__ == "__main__":
    main()
