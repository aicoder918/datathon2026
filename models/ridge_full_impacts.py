"""Ridge submissions using full-train event impacts on train and test.

This intentionally removes the train/test mismatch where train rows use OOF
event-impact columns while test rows use impacts fit on all training data.

Outputs:
  - submissions/chatgpt/ridge_fullimpacts_a3000.csv
  - submissions/chatgpt/ridge_fullimpacts_strong.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from features import (
    build_event_features_multi,
    build_event_features_sector_multi,
    fit_template_impacts_multi,
    fit_template_impacts_sector_multi,
    load_test,
    load_train_base,
    shape_positions,
    finalize,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_Q = 0.35
RIDGE_ALPHAS_STRONG = np.logspace(1, 6, 16)


def fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    kind: str,
) -> tuple[np.ndarray, float]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.to_numpy(dtype=np.float64))
    Xte = scaler.transform(X_test.to_numpy(dtype=np.float64))
    if kind == "a3000":
        model = Ridge(alpha=3000.0)
        model.fit(Xtr, y_train.to_numpy(dtype=np.float64))
        return np.asarray(model.predict(Xte), dtype=float), 3000.0
    if kind == "strong":
        model = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
        model.fit(Xtr, y_train.to_numpy(dtype=np.float64))
        return np.asarray(model.predict(Xte), dtype=float), float(model.alpha_)
    raise ValueError(kind)


def save_submission(sessions: np.ndarray, pred: np.ndarray, test_vol: np.ndarray, name: str) -> None:
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({"session": sessions.astype(int), "target_position": pos})
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())


X_base, y_train, headlines_train, bars_train = load_train_base()
impacts = fit_template_impacts_multi(headlines_train, bars_train)
sec_impacts = fit_template_impacts_sector_multi(headlines_train, bars_train)
train_sessions = X_base.index.to_numpy()

X_train = X_base.join(build_event_features_multi(headlines_train, train_sessions, impacts))
X_train = X_train.join(build_event_features_sector_multi(headlines_train, train_sessions, sec_impacts))
X_test = load_test(impacts=impacts)

print(f"Train: {X_train.shape}   Test: {X_test.shape}")
test_vol = np.asarray(X_test["vol"].values, dtype=float)
sessions = X_test.index.to_numpy()

for kind, name in [
    ("a3000", "ridge_fullimpacts_a3000.csv"),
    ("strong", "ridge_fullimpacts_strong.csv"),
]:
    pred, alpha = fit_predict(X_train, y_train, X_test, kind)
    print(f"{name:<28s} alpha={alpha:.6f} pred mean={pred.mean():+.5f} std={pred.std():.5f}")
    save_submission(sessions, pred, test_vol, name)
