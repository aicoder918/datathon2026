"""Template + direction + volatility baseline: per-session position
= 1 + alpha * (z_template + beta * z_direction + gamma * z_vol). No model.

- Template: sum of train-estimated impacts across session headlines.
- Direction: close[49]/close[0] - 1 (mean-reverts against the 50->99 target).
- Vol:      std of 1-bar returns over bars 0..49.

Submission format: session, target_position
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_model.data_seq import (
    compute_template_impacts,
    load_bars,
    load_headlines,
)


def session_template_score(hls_df: pd.DataFrame, template_impacts: np.ndarray) -> dict:
    """Sum of template_impacts over headlines firing in each session."""
    n = len(template_impacts)
    out = {}
    for s, g in hls_df.groupby("session"):
        tids = g["template_index"].to_numpy(dtype=int)
        tids = tids[(tids >= 0) & (tids < n)]
        out[int(s)] = float(template_impacts[tids].sum()) if len(tids) else 0.0
    return out


def session_direction_0_49(bars_df: pd.DataFrame) -> dict:
    """ret_0_49[s] = close[49]/close[0] - 1."""
    out = {}
    for s, g in bars_df.groupby("session"):
        g = g.sort_values("bar_ix")
        c = g["close"].to_numpy(dtype=float)
        if len(c) < 50:
            out[int(s)] = 0.0
            continue
        out[int(s)] = float(c[49] / (c[0] + 1e-8) - 1.0)
    return out


def session_vol_0_49(bars_df: pd.DataFrame) -> dict:
    """vol[s] = std of 1-bar returns over bars 0..49."""
    out = {}
    for s, g in bars_df.groupby("session"):
        g = g.sort_values("bar_ix").head(50)
        c = g["close"].to_numpy(dtype=float)
        if len(c) < 2:
            out[int(s)] = 0.0
            continue
        ret = c[1:] / (c[:-1] + 1e-8) - 1.0
        out[int(s)] = float(np.std(ret))
    return out


def sharpe(pos, ret) -> float:
    pos = np.asarray(pos, dtype=float)
    ret = np.asarray(ret, dtype=float)
    pnl = pos * ret
    std = pnl.std()
    return 0.0 if std < 1e-12 else float(pnl.mean() / std * 16)


def train_targets(bars_full: pd.DataFrame) -> dict:
    out = {}
    for s, g in bars_full.groupby("session"):
        g = g.sort_values("bar_ix")
        if len(g) < 100:
            continue
        c = g["close"].to_numpy()
        out[int(s)] = float(c[99] / (c[49] + 1e-8) - 1.0)
    return out


def _z(arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (arr - mu) / (sigma + 1e-12)


def eval_on_train(template_impacts: np.ndarray):
    """In-sample train eval + 3D sweep over (alpha, beta, gamma)."""
    bars = load_bars("train_full")
    hls = load_headlines("train")
    tpl_scores = session_template_score(hls, template_impacts)
    dir_scores = session_direction_0_49(bars)
    vol_scores = session_vol_0_49(bars)
    y_map = train_targets(bars)

    sessions = sorted(y_map.keys())
    y = np.array([y_map[s] for s in sessions])
    tpl = np.array([tpl_scores.get(s, 0.0) for s in sessions])
    dir_ = np.array([dir_scores.get(s, 0.0) for s in sessions])
    vol = np.array([vol_scores.get(s, 0.0) for s in sessions])

    tpl_mu, tpl_sigma = float(tpl.mean()), float(tpl.std())
    dir_mu, dir_sigma = float(dir_.mean()), float(dir_.std())
    vol_mu, vol_sigma = float(vol.mean()), float(vol.std())
    z_tpl = _z(tpl, tpl_mu, tpl_sigma)
    z_dir = _z(dir_, dir_mu, dir_sigma)
    z_vol = _z(vol, vol_mu, vol_sigma)

    print(f"Train sessions: {len(sessions)}")
    print(f"  target    mean={y.mean():+.5f}, std={y.std():.5f}")
    print(f"  template  mu={tpl_mu:+.6f}, std={tpl_sigma:.6f}")
    print(f"  dir 0-49  mu={dir_mu:+.6f}, std={dir_sigma:.6f}")
    print(f"  vol 0-49  mu={vol_mu:+.6f}, std={vol_sigma:.6f}")
    print(f"\n  Feature correlations with target y:")
    print(f"    corr(template, y) = {np.corrcoef(tpl, y)[0, 1]:+.4f}")
    print(f"    corr(dir,      y) = {np.corrcoef(dir_, y)[0, 1]:+.4f}")
    print(f"    corr(vol,      y) = {np.corrcoef(vol, y)[0, 1]:+.4f}")
    print(f"  Feature cross-correlations:")
    print(f"    corr(template, dir) = {np.corrcoef(tpl, dir_)[0, 1]:+.4f}")
    print(f"    corr(template, vol) = {np.corrcoef(tpl, vol)[0, 1]:+.4f}")
    print(f"    corr(dir,      vol) = {np.corrcoef(dir_, vol)[0, 1]:+.4f}")

    print(f"\n  constant 1.0             : {sharpe(np.ones_like(y), y):.3f}")

    alpha_grid = [0.1, 0.25, 0.5, 1.0]
    beta_grid = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5]
    gamma_grid = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    best = (-np.inf, None)
    for alpha, beta, gamma in product(alpha_grid, beta_grid, gamma_grid):
        combined = z_tpl + beta * z_dir + gamma * z_vol
        pos = 1.0 + alpha * combined
        s = sharpe(pos, y)
        if s > best[0]:
            best = (s, (alpha, beta, gamma))

    print("\n  ─ top-10 (alpha, beta, gamma) combos by in-sample Sharpe ─")
    scored = []
    for alpha, beta, gamma in product(alpha_grid, beta_grid, gamma_grid):
        combined = z_tpl + beta * z_dir + gamma * z_vol
        pos = 1.0 + alpha * combined
        scored.append((sharpe(pos, y), alpha, beta, gamma))
    scored.sort(reverse=True)
    for s, a, b, g in scored[:10]:
        print(f"    alpha={a:.2f}, beta={b:+.2f}, gamma={g:+.2f}  ->  Sharpe={s:.3f}")

    print(f"\n  best in-sample: alpha={best[1][0]}, beta={best[1][1]}, "
          f"gamma={best[1][2]}, Sharpe={best[0]:.3f}")

    # Also report template-only and template+direction for reference.
    for a_ref in [0.25, 0.5]:
        s_t = sharpe(1.0 + a_ref * z_tpl, y)
        s_td = sharpe(1.0 + a_ref * (z_tpl - z_dir), y)
        print(f"  reference:   alpha={a_ref}  "
              f"tpl-only={s_t:.3f},  tpl+dir(beta=-1)={s_td:.3f}")

    return {
        "tpl_mu": tpl_mu, "tpl_sigma": tpl_sigma,
        "dir_mu": dir_mu, "dir_sigma": dir_sigma,
        "vol_mu": vol_mu, "vol_sigma": vol_sigma,
        "best_alpha": best[1][0],
        "best_beta": best[1][1],
        "best_gamma": best[1][2],
        "best_sharpe": best[0],
    }


def build_submission(template_impacts: np.ndarray, stats: dict,
                     alpha: float, beta: float, gamma: float,
                     out_path: Path) -> None:
    parts = []
    for split in ("public", "private"):
        bars = load_bars(split)
        hls = load_headlines(split)
        tpl_scores = session_template_score(hls, template_impacts)
        dir_scores = session_direction_0_49(bars)
        vol_scores = session_vol_0_49(bars)

        sessions = sorted(int(s) for s in bars["session"].unique())
        tpl = np.array([tpl_scores.get(s, 0.0) for s in sessions])
        dir_ = np.array([dir_scores.get(s, 0.0) for s in sessions])
        vol = np.array([vol_scores.get(s, 0.0) for s in sessions])

        z_tpl = _z(tpl, stats["tpl_mu"], stats["tpl_sigma"])
        z_dir = _z(dir_, stats["dir_mu"], stats["dir_sigma"])
        z_vol = _z(vol, stats["vol_mu"], stats["vol_sigma"])
        combined = z_tpl + beta * z_dir + gamma * z_vol
        pos = 1.0 + alpha * combined

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
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission ({len(sub)} rows) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="submission_template_dir_vol.csv")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha (default: pick from train sweep)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override beta (default: pick from train sweep)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override gamma (default: pick from train sweep)")
    args = parser.parse_args()

    print("Computing template impacts (train-only)...")
    bars_full = load_bars("train_full")
    hls_train = load_headlines("train")
    template_impacts, counts = compute_template_impacts(bars_full, hls_train)
    print(f"  impacts: min={template_impacts.min():+.5f} "
          f"max={template_impacts.max():+.5f} "
          f"mean={template_impacts.mean():+.5f} "
          f"(non-zero templates: {int((counts > 0).sum())})")

    print("\n=== Train in-sample evaluation ===")
    stats = eval_on_train(template_impacts)

    alpha = args.alpha if args.alpha is not None else stats["best_alpha"]
    beta = args.beta if args.beta is not None else stats["best_beta"]
    gamma = args.gamma if args.gamma is not None else stats["best_gamma"]
    print(f"\nBuilding submission: alpha={alpha}, beta={beta}, gamma={gamma}")
    build_submission(template_impacts, stats, alpha, beta, gamma, Path(args.out))


if __name__ == "__main__":
    main()
