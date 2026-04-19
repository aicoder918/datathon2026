from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(MODELS_DIR))

from features import DATA_DIR, extract_event, sharpe  # noqa: E402

K_FORWARD = 5


def zscore_with_train_stats(values: pd.Series, mean: float, std: float) -> pd.Series:
    denom = std if std > 1e-12 else 1.0
    return (values - mean) / denom


def compute_session_baseline(bars_full: pd.DataFrame) -> pd.Series:
    bars = bars_full.sort_values(["session", "bar_ix"]).copy()
    out_rows: list[tuple[int, float]] = []
    for session, g in bars.groupby("session", sort=True):
        close = g["close"].to_numpy(dtype=float)
        seen_n = min(50, len(close))
        local = []
        for t in range(seen_n):
            end = min(t + K_FORWARD, len(close) - 1)
            local.append(close[end] / close[t] - 1.0)
        out_rows.append((int(session), float(np.mean(local)) if local else 0.0))
    return pd.Series(dict(out_rows), name="session_baseline").sort_index()


def compute_template_impacts(bars_full: pd.DataFrame, hls_train: pd.DataFrame) -> pd.Series:
    bars = bars_full[["session", "bar_ix", "close"]].set_index(["session", "bar_ix"])["close"]
    max_bar = bars_full.groupby("session")["bar_ix"].max()
    session_baseline = compute_session_baseline(bars_full)

    h = hls_train.copy()
    h["_tid"] = [extract_event(headline)[0] for headline in h["headline"]]
    h = h[h["_tid"] >= 0].copy()

    starts = list(zip(h["session"], h["bar_ix"]))
    ends = np.minimum(h["bar_ix"].to_numpy() + K_FORWARD, h["session"].map(max_bar).to_numpy())
    h["c0"] = bars.reindex(starts).to_numpy()
    h["c1"] = bars.reindex(list(zip(h["session"], ends))).to_numpy()
    h["local_fwd"] = h["c1"] / h["c0"] - 1.0
    h["session_baseline"] = h["session"].map(session_baseline)
    h["template_effect"] = h["local_fwd"] - h["session_baseline"]

    impacts = h.groupby("_tid")["template_effect"].mean().sort_index()
    return impacts


def session_template_score(hls: pd.DataFrame, impacts: pd.Series, all_sessions: np.ndarray) -> pd.Series:
    h = hls.copy()
    h["_tid"] = [extract_event(headline)[0] for headline in h["headline"]]
    h["impact"] = h["_tid"].map(impacts).fillna(0.0)
    score = h.groupby("session")["impact"].sum()
    out = pd.Series(0.0, index=all_sessions, name="template_score")
    out.loc[score.index] = score.to_numpy(dtype=float)
    return out.sort_index()


def session_direction_0_49(bars_seen: pd.DataFrame) -> pd.Series:
    g = bars_seen.sort_values(["session", "bar_ix"]).groupby("session")
    first = g["close"].first()
    last = g["close"].last()
    return (last / first - 1.0).rename("direction_0_49").sort_index()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars_seen_train = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    bars_unseen_train = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    hls_train = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")

    bars_seen_test = pd.concat([
        pd.read_parquet(DATA_DIR / "bars_seen_public_test.parquet"),
        pd.read_parquet(DATA_DIR / "bars_seen_private_test.parquet"),
    ], ignore_index=True)
    hls_test = pd.concat([
        pd.read_parquet(DATA_DIR / "headlines_seen_public_test.parquet"),
        pd.read_parquet(DATA_DIR / "headlines_seen_private_test.parquet"),
    ], ignore_index=True)

    return bars_seen_train, bars_unseen_train, hls_train, bars_seen_test, hls_test


def build_train_features(
    bars_seen_train: pd.DataFrame,
    bars_unseen_train: pd.DataFrame,
    hls_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    bars_full = pd.concat([bars_seen_train, bars_unseen_train], ignore_index=True)
    sessions = np.sort(bars_seen_train["session"].unique())
    impacts = compute_template_impacts(bars_full, hls_train)
    template_score = session_template_score(hls_train, impacts, sessions)
    direction = session_direction_0_49(bars_seen_train).reindex(sessions).fillna(0.0)

    target = (
        bars_unseen_train.sort_values("bar_ix").groupby("session")["close"].last()
        / bars_seen_train.sort_values("bar_ix").groupby("session")["close"].last()
        - 1.0
    ).rename("y").reindex(sessions)

    X = pd.DataFrame({
        "template_score": template_score,
        "direction_0_49": direction,
    }, index=sessions)
    return X, target, impacts


def build_test_features(
    bars_seen_test: pd.DataFrame,
    hls_test: pd.DataFrame,
    impacts: pd.Series,
) -> pd.DataFrame:
    sessions = np.sort(bars_seen_test["session"].unique())
    template_score = session_template_score(hls_test, impacts, sessions)
    direction = session_direction_0_49(bars_seen_test).reindex(sessions).fillna(0.0)
    return pd.DataFrame({
        "template_score": template_score,
        "direction_0_49": direction,
    }, index=sessions)


def make_positions(
    z_template: pd.Series,
    z_direction: pd.Series,
    alpha: float,
    beta: float,
) -> pd.Series:
    return 1.0 + alpha * (z_template + beta * z_direction)


def sweep_alpha_beta(
    z_template: pd.Series,
    z_direction: pd.Series,
    target: pd.Series,
) -> tuple[float, float, float]:
    alpha_grid = [0.10, 0.25, 0.50, 0.75, 1.00]
    beta_grid = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    best = (-np.inf, 0.5, -1.0)
    y = target.to_numpy(dtype=float)
    for alpha in alpha_grid:
        for beta in beta_grid:
            pos = make_positions(z_template, z_direction, alpha, beta).to_numpy(dtype=float)
            s = sharpe(pos, y)
            if s > best[0]:
                best = (s, alpha, beta)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()

    bars_seen_train, bars_unseen_train, hls_train, bars_seen_test, hls_test = load_data()
    X_train, y_train, impacts = build_train_features(bars_seen_train, bars_unseen_train, hls_train)
    X_test = build_test_features(bars_seen_test, hls_test, impacts)

    mu_t = float(X_train["template_score"].mean())
    sd_t = float(X_train["template_score"].std(ddof=0))
    mu_d = float(X_train["direction_0_49"].mean())
    sd_d = float(X_train["direction_0_49"].std(ddof=0))

    z_template_train = zscore_with_train_stats(X_train["template_score"], mu_t, sd_t)
    z_direction_train = zscore_with_train_stats(X_train["direction_0_49"], mu_d, sd_d)

    if args.alpha is None or args.beta is None:
        best_sharpe, alpha, beta = sweep_alpha_beta(z_template_train, z_direction_train, y_train)
        print(f"best in-sample sharpe={best_sharpe:.3f} alpha={alpha} beta={beta}")
    else:
        alpha = float(args.alpha)
        beta = float(args.beta)
        pos_train = make_positions(z_template_train, z_direction_train, alpha, beta)
        print(f"forced params in-sample sharpe={sharpe(pos_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float)):.3f}")

    corr_template = float(np.corrcoef(X_train["template_score"], y_train)[0, 1])
    corr_direction = float(np.corrcoef(X_train["direction_0_49"], y_train)[0, 1])
    corr_inter = float(np.corrcoef(X_train["template_score"] * X_train["direction_0_49"], y_train)[0, 1])
    print(f"corr(template, y)={corr_template:+.3f}")
    print(f"corr(direction, y)={corr_direction:+.3f}")
    print(f"corr(template*direction, y)={corr_inter:+.3f}")

    z_template_test = zscore_with_train_stats(X_test["template_score"], mu_t, sd_t)
    z_direction_test = zscore_with_train_stats(X_test["direction_0_49"], mu_d, sd_d)
    positions = make_positions(z_template_test, z_direction_test, alpha, beta)

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
