"""Compact, ridge-friendly replacements for headline and event feature blocks.

The idea is to reduce redundancy inside the current sentiment/event aggregates
instead of adding richer features. We keep the bar block unchanged and replace:

- 9 headline features -> 5 compact headline features
- 12 event-impact features -> 4 compact event features

Outputs:
  - submissions/chatgpt/ridge_compact_headline.csv
  - submissions/chatgpt/ridge_compact_event.csv
  - submissions/chatgpt/ridge_compact_all.csv
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
THRESHOLD_Q = 0.35

HL_COLS = [
    "hl_n",
    "hl_n_recent",
    "hl_last_bar",
    "hl_mean_bar",
    "hl_net_sent",
    "hl_net_sent_recent",
    "hl_mean_sent",
    "hl_n_pos",
    "hl_n_neg",
]
EVT_COLS = [f"event_impact_k{k}" for k in (3, 5, 10)] + \
           [f"event_impact_recent_k{k}" for k in (3, 5, 10)] + \
           [f"event_impact_sec_k{k}" for k in (3, 5, 10)] + \
           [f"event_impact_sec_recent_k{k}" for k in (3, 5, 10)]


def compact_headline_features(X: pd.DataFrame) -> pd.DataFrame:
    n = X["hl_n"].astype(float)
    denom = np.maximum(n.to_numpy(dtype=float), 1.0)
    out = pd.DataFrame(index=X.index)
    out["hl_n"] = X["hl_n"]
    out["hl_last_bar"] = X["hl_last_bar"]
    out["hl_sent_sum"] = X["hl_net_sent"]
    out["hl_sent_avg"] = X["hl_net_sent"].to_numpy(dtype=float) / denom
    out["hl_pos_neg_balance"] = (
        X["hl_n_pos"].to_numpy(dtype=float) - X["hl_n_neg"].to_numpy(dtype=float)
    ) / denom
    return out


def compact_event_features(X: pd.DataFrame) -> pd.DataFrame:
    tid = X[[f"event_impact_k{k}" for k in (3, 5, 10)]].mean(axis=1)
    tid_recent = X[[f"event_impact_recent_k{k}" for k in (3, 5, 10)]].mean(axis=1)
    sec = X[[f"event_impact_sec_k{k}" for k in (3, 5, 10)]].mean(axis=1)
    sec_recent = X[[f"event_impact_sec_recent_k{k}" for k in (3, 5, 10)]].mean(axis=1)
    out = pd.DataFrame(index=X.index)
    out["evt_tid_mean"] = tid
    out["evt_tid_recent_lift"] = tid_recent - tid
    out["evt_sec_lift"] = sec - tid
    out["evt_sec_recent_lift"] = sec_recent - sec
    return out


def fit_predict(
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


def save_submission(
    sessions: np.ndarray,
    pred: np.ndarray,
    test_vol: np.ndarray,
    name: str,
) -> None:
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": pos,
    })
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())


X_train_full, y_train = load_train()
X_test_full = load_test()
test_vol = np.asarray(X_test_full["vol"].values, dtype=float)
sessions = X_test_full.index.to_numpy()

headline_train = compact_headline_features(X_train_full)
headline_test = compact_headline_features(X_test_full)
event_train = compact_event_features(X_train_full)
event_test = compact_event_features(X_test_full)

variants = {
    "ridge_compact_headline.csv": (
        X_train_full.drop(columns=HL_COLS).join(headline_train),
        X_test_full.drop(columns=HL_COLS).join(headline_test),
    ),
    "ridge_compact_event.csv": (
        X_train_full.drop(columns=EVT_COLS).join(event_train),
        X_test_full.drop(columns=EVT_COLS).join(event_test),
    ),
    "ridge_compact_all.csv": (
        X_train_full.drop(columns=HL_COLS + EVT_COLS).join(headline_train).join(event_train),
        X_test_full.drop(columns=HL_COLS + EVT_COLS).join(headline_test).join(event_test),
    ),
}

print(f"Train: {X_train_full.shape}   Test: {X_test_full.shape}")
for name, (Xtr, Xte) in variants.items():
    pred, alpha = fit_predict(Xtr, y_train, Xte)
    print(
        f"{name:<26s} alpha={alpha:.6f} ncols={Xtr.shape[1]} "
        f"pred mean={pred.mean():+.5f} std={pred.std():.5f}"
    )
    save_submission(sessions, pred, test_vol, name)
