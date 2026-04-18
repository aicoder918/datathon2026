"""Reverse-engineer latent simulator parameters from the first 50 bars.

Instead of using the full production feature stack, this script estimates a
small set of generator-like session parameters from the visible half-session:
drift, volatility, persistence, jumpiness, trend ratio, and within-session news
response. A strong ridge then maps those latent estimates to forward return.

Output:
  - submissions/chatgpt/simulator_params_ridge.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from features import DATA_DIR, SENT_MAP, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHAS = np.logspace(1, 6, 16)
THRESHOLD_Q = 0.35


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    x0 = x - x.mean()
    denom = float(np.dot(x0, x0))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x0, y - y.mean()) / denom)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    xs = x.std()
    ys = y.std()
    if xs <= 1e-12 or ys <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_session_params(seen_bars: pd.DataFrame, headlines: pd.DataFrame) -> pd.DataFrame:
    bars = seen_bars.sort_values(["session", "bar_ix"]).copy()
    bars["log_close"] = np.log(np.maximum(bars["close"], 1e-12))
    bars["log_ret"] = bars.groupby("session")["log_close"].diff().fillna(0.0)
    bars["bar_range"] = (bars["high"] - bars["low"]) / np.maximum(bars["close"], 1e-12)

    h = headlines.merge(SENT_MAP, left_on="headline", right_index=True, how="left")
    h["signed"] = h["signed"].fillna(0.0)
    sent_bar = (
        h.groupby(["session", "bar_ix"])["signed"]
        .sum()
        .unstack()
        .fillna(0.0)
        .reindex(columns=range(50), fill_value=0.0)
    )
    cnt_bar = (
        h.groupby(["session", "bar_ix"])
        .size()
        .unstack()
        .fillna(0.0)
        .reindex(columns=range(50), fill_value=0.0)
    )

    rows: list[dict[str, float]] = []
    for session, g in bars.groupby("session", sort=True):
        close = g["close"].to_numpy(dtype=float)
        log_close = g["log_close"].to_numpy(dtype=float)
        ret = g["log_ret"].to_numpy(dtype=float)[1:]
        bar_range = g["bar_range"].to_numpy(dtype=float)

        mu = float(ret.mean()) if len(ret) else 0.0
        sigma = float(ret.std()) if len(ret) else 0.0
        abs_sum = float(np.abs(ret).sum()) if len(ret) else 0.0
        trend_ratio = float(ret.sum() / abs_sum) if abs_sum > 1e-12 else 0.0
        drift_t = float(mu / sigma * np.sqrt(len(ret))) if sigma > 1e-12 and len(ret) else 0.0
        jump_frac = float(np.mean(np.abs(ret - mu) > 2.0 * sigma)) if sigma > 1e-12 and len(ret) else 0.0

        ar1_ret = safe_slope(ret[:-1], ret[1:]) if len(ret) >= 3 else 0.0
        price_dev = log_close - log_close.mean()
        ar1_price = safe_slope(price_dev[:-1], price_dev[1:]) if len(price_dev) >= 3 else 0.0

        sent = sent_bar.loc[session].to_numpy(dtype=float) if session in sent_bar.index else np.zeros(50, dtype=float)
        cnt = cnt_bar.loc[session].to_numpy(dtype=float) if session in cnt_bar.index else np.zeros(50, dtype=float)
        next_ret = np.diff(log_close)
        news_beta = safe_slope(sent[:-1], next_ret) if len(next_ret) else 0.0
        news_corr = safe_corr(sent[:-1], next_ret) if len(next_ret) else 0.0
        recent_news_share = float(cnt[40:].sum() / max(cnt.sum(), 1.0))

        rows.append({
            "session": float(session),
            "mu_hat": mu,
            "sigma_hat": sigma,
            "drift_tstat": drift_t,
            "trend_ratio": trend_ratio,
            "ar1_ret": ar1_ret,
            "ar1_price_dev": ar1_price,
            "jump_frac": jump_frac,
            "range_mean": float(bar_range.mean()),
            "range_std": float(bar_range.std()),
            "seen_return": float(close[-1] / close[0] - 1.0),
            "news_beta": news_beta,
            "news_corr": news_corr,
            "news_density": float(cnt.sum() / 50.0),
            "recent_news_share": recent_news_share,
            "sent_total": float(sent.sum()),
            "sent_abs_total": float(np.abs(sent).sum()),
        })

    out = pd.DataFrame(rows).set_index("session").sort_index()
    out.index = out.index.astype(int)
    return out.fillna(0.0)


def load_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    seen_train = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    unseen_train = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    headlines_train = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")

    seen_test = pd.concat([
        pd.read_parquet(DATA_DIR / "bars_seen_public_test.parquet"),
        pd.read_parquet(DATA_DIR / "bars_seen_private_test.parquet"),
    ], ignore_index=True)
    headlines_test = pd.concat([
        pd.read_parquet(DATA_DIR / "headlines_seen_public_test.parquet"),
        pd.read_parquet(DATA_DIR / "headlines_seen_private_test.parquet"),
    ], ignore_index=True)

    last_seen = seen_train.sort_values("bar_ix").groupby("session")["close"].last()
    last_unseen = unseen_train.sort_values("bar_ix").groupby("session")["close"].last()
    y = (last_unseen / last_seen - 1).rename("y")

    X_train = build_session_params(seen_train, headlines_train)
    X_test = build_session_params(seen_test, headlines_test)
    return X_train, y.reindex(X_train.index), X_test


X_train_df, y_train_s, X_test_df = load_train_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_df.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)

model = RidgeCV(alphas=RIDGE_ALPHAS)
model.fit(Xtr, y_train)
pred = np.asarray(model.predict(Xte), dtype=float)
lo, hi = np.quantile(pred, [0.005, 0.995])
pred = np.clip(pred, lo, hi)
print(f"ridge alpha={float(model.alpha_):.6f} pred mean={pred.mean():+.5f} std={pred.std():.5f}")

test_vol = X_test_df["sigma_hat"].to_numpy(dtype=float)
pos = shape_positions(pred, test_vol, "thresholded", threshold_q=THRESHOLD_Q)
pos = finalize(pos)

submission = pd.DataFrame({
    "session": X_test_df.index.astype(int),
    "target_position": pos,
})
out_path = OUT_DIR / "simulator_params_ridge.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved submission: {out_path}")
print(submission["target_position"].describe().to_string())
