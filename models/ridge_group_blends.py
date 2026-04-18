"""Groupwise ridge models and simple block blends.

Uses three disjoint feature groups from the production 40-column matrix:
  - bars
  - headlines
  - event-impact

Generates standalone submissions for each block plus two simple blends:
  - ridge_bars_only.csv
  - ridge_headline_only.csv
  - ridge_event_only.csv
  - ridge_bars_event_blend.csv
  - ridge_bars_headline_event_blend.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHAS = np.logspace(1, 6, 16)
BEST_KIND = "thresholded_inv_vol"
THRESHOLD_Q = 0.35


def fit_ridge_block(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.to_numpy(dtype=np.float64))
    Xte = scaler.transform(X_test.to_numpy(dtype=np.float64))
    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(Xtr, y_train.to_numpy(dtype=np.float64))
    pred = np.asarray(model.predict(Xte), dtype=float)
    return pred, float(model.alpha_)


def pred_to_positions(pred: np.ndarray, test_vol: np.ndarray) -> np.ndarray:
    pos = shape_positions(pred, test_vol, BEST_KIND, threshold_q=THRESHOLD_Q)
    return finalize(pos)


def save_submission(
    sessions: np.ndarray,
    positions: np.ndarray,
    name: str,
) -> None:
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": positions,
    })
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission.describe())


X_train, y_train = load_train()
X_test = load_test()
sessions = X_test.index.to_numpy()
test_vol = np.asarray(X_test["vol"].values, dtype=float)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

bar_cols = [
    "close_first", "close_last", "seen_ret", "mom_1", "mom_3", "mom_5", "mom_10",
    "vol", "vol_recent", "max_high", "min_low", "dist_to_high", "dist_to_low",
    "body_mean", "range_mean", "close_pos_mean", "close_pos_last", "max_drawdown", "ret_skew",
]
hl_cols = [c for c in X_train.columns if c.startswith("hl_")]
evt_cols = [c for c in X_train.columns if c.startswith("event_impact")]

preds: dict[str, np.ndarray] = {}
for name, cols in [("bars", bar_cols), ("headline", hl_cols), ("event", evt_cols)]:
    pred, alpha = fit_ridge_block(X_train[cols], y_train, X_test[cols])
    preds[name] = pred
    print(
        f"{name:>8} alpha={alpha:.6f} ncols={len(cols)} "
        f"pred mean={pred.mean():+.5f} std={pred.std():.5f}"
    )

save_submission(sessions, pred_to_positions(preds["bars"], test_vol), "ridge_bars_only.csv")
save_submission(sessions, pred_to_positions(preds["headline"], test_vol), "ridge_headline_only.csv")
save_submission(sessions, pred_to_positions(preds["event"], test_vol), "ridge_event_only.csv")

bars_event = 0.5 * preds["bars"] + 0.5 * preds["event"]
save_submission(
    sessions,
    pred_to_positions(bars_event, test_vol),
    "ridge_bars_event_blend.csv",
)

all_group_blend = 0.5 * preds["bars"] + 0.25 * preds["headline"] + 0.25 * preds["event"]
save_submission(
    sessions,
    pred_to_positions(all_group_blend, test_vol),
    "ridge_bars_headline_event_blend.csv",
)
