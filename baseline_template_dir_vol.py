from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import baseline_template as bt  # noqa: E402


def realized_vol_0_49(bars_seen: pd.DataFrame, kind: str) -> pd.Series:
    bars = bars_seen.sort_values(["session", "bar_ix"])
    log_ret = bars.groupby("session")["close"].apply(
        lambda s: pd.Series(np.diff(np.log(s.to_numpy(dtype=float))))
    ).reset_index(level=0, drop=True)
    if kind == "std":
        vol = log_ret.groupby(bars.iloc[1:].set_index("session", append=False).index).std()
    else:
        vol = bars.groupby("session")["close"].apply(
            lambda s: np.diff(np.log(s.to_numpy(dtype=float))).std() if len(s) > 1 else 0.0
        )
        if kind == "log_std":
            vol = np.log(np.maximum(vol, 1e-8))
    return pd.Series(vol, name=f"vol_{kind}").sort_index()


def session_vol_0_49(bars_seen: pd.DataFrame, kind: str) -> pd.Series:
    vol = bars_seen.sort_values(["session", "bar_ix"]).groupby("session")["close"].apply(
        lambda s: np.diff(np.log(s.to_numpy(dtype=float))).std() if len(s) > 1 else 0.0
    )
    if kind == "log_std":
        vol = np.log(np.maximum(vol, 1e-8))
    return vol.rename(f"vol_{kind}").sort_index()


def make_positions(
    z_template: pd.Series,
    z_direction: pd.Series,
    z_vol: pd.Series,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.Series:
    return 1.0 + alpha * (z_template + beta * z_direction + gamma * z_vol)


def sweep_alpha_beta_gamma(
    z_template: pd.Series,
    z_direction: pd.Series,
    z_vol: pd.Series,
    target: pd.Series,
) -> tuple[float, float, float, float, list[tuple[float, float, float, float]]]:
    alpha_grid = [0.35, 0.50, 0.65, 0.80]
    beta_grid = [-1.25, -1.00, -0.85, -0.70]
    gamma_grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    ranked: list[tuple[float, float, float, float]] = []
    y = target.to_numpy(dtype=float)
    for alpha in alpha_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                pos = make_positions(z_template, z_direction, z_vol, alpha, beta, gamma).to_numpy(dtype=float)
                s = bt.sharpe(pos, y)
                ranked.append((s, alpha, beta, gamma))
    ranked.sort(reverse=True)
    best = ranked[0]
    return best[0], best[1], best[2], best[3], ranked[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--vol-kind", choices=["std", "log_std"], default="std")
    args = parser.parse_args()

    bars_seen_train, bars_unseen_train, hls_train, bars_seen_test, hls_test = bt.load_data()
    X_train, y_train, impacts = bt.build_train_features(bars_seen_train, bars_unseen_train, hls_train)
    X_test = bt.build_test_features(bars_seen_test, hls_test, impacts)

    train_vol = session_vol_0_49(bars_seen_train, args.vol_kind).reindex(X_train.index).fillna(0.0)
    test_vol = session_vol_0_49(bars_seen_test, args.vol_kind).reindex(X_test.index).fillna(0.0)

    mu_t = float(X_train["template_score"].mean())
    sd_t = float(X_train["template_score"].std(ddof=0))
    mu_d = float(X_train["direction_0_49"].mean())
    sd_d = float(X_train["direction_0_49"].std(ddof=0))
    mu_v = float(train_vol.mean())
    sd_v = float(train_vol.std(ddof=0))

    z_template_train = bt.zscore_with_train_stats(X_train["template_score"], mu_t, sd_t)
    z_direction_train = bt.zscore_with_train_stats(X_train["direction_0_49"], mu_d, sd_d)
    z_vol_train = bt.zscore_with_train_stats(train_vol, mu_v, sd_v)

    if args.alpha is None or args.beta is None or args.gamma is None:
        best_sharpe, alpha, beta, gamma, top10 = sweep_alpha_beta_gamma(
            z_template_train, z_direction_train, z_vol_train, y_train
        )
        print(f"best in-sample sharpe={best_sharpe:.3f} alpha={alpha} beta={beta} gamma={gamma} vol_kind={args.vol_kind}")
        print("top candidates:")
        for s, a, b, g in top10[:5]:
            print(f"  sharpe={s:.3f} alpha={a} beta={b} gamma={g}")
    else:
        alpha = float(args.alpha)
        beta = float(args.beta)
        gamma = float(args.gamma)
        pos_train = make_positions(z_template_train, z_direction_train, z_vol_train, alpha, beta, gamma)
        score = bt.sharpe(pos_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))
        print(f"forced params in-sample sharpe={score:.3f} alpha={alpha} beta={beta} gamma={gamma} vol_kind={args.vol_kind}")

    z_template_test = bt.zscore_with_train_stats(X_test["template_score"], mu_t, sd_t)
    z_direction_test = bt.zscore_with_train_stats(X_test["direction_0_49"], mu_d, sd_d)
    z_vol_test = bt.zscore_with_train_stats(test_vol, mu_v, sd_v)
    positions = make_positions(z_template_test, z_direction_test, z_vol_test, alpha, beta, gamma)

    submission = pd.DataFrame({
        "session": X_test.index.astype(int),
        "target_position": positions.to_numpy(dtype=float),
    }).sort_values("session")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(submission["target_position"].describe().to_string())


if __name__ == "__main__":
    main()
