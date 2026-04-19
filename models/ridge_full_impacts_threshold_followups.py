"""Adaptive follow-up around the improving full-impact + lower-threshold branch.

Outputs:
  - submissions/chatgpt/ridge_fullimpacts_strong_tq25.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq28.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq32.csv
  - submissions/chatgpt/ridge_fullimpacts_a2000_tq30.csv
  - submissions/chatgpt/ridge_fullimpacts_a2000_tq25.csv
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
RIDGE_ALPHAS_STRONG = np.logspace(1, 6, 16)


def load_fullimpact_matrix() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_base, y_train, headlines_train, bars_train = load_train_base()
    impacts = fit_template_impacts_multi(headlines_train, bars_train)
    sec_impacts = fit_template_impacts_sector_multi(headlines_train, bars_train)
    train_sessions = X_base.index.to_numpy()
    X_train = X_base.join(build_event_features_multi(headlines_train, train_sessions, impacts))
    X_train = X_train.join(build_event_features_sector_multi(headlines_train, train_sessions, sec_impacts))
    X_test = load_test(impacts=impacts)
    return X_train, y_train, X_test


def save_submission(sessions: np.ndarray, pred: np.ndarray, vol: np.ndarray, tq: float, name: str) -> None:
    pos = shape_positions(pred, vol, "thresholded_inv_vol", threshold_q=tq)
    pos = finalize(pos)
    submission = pd.DataFrame({"session": sessions.astype(int), "target_position": pos})
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())


X_train_df, y_train_s, X_test_df = load_fullimpact_matrix()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_df.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

strong = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
strong.fit(Xtr, y_train)
pred_strong = np.asarray(strong.predict(Xte), dtype=float)
print(f"strong alpha={float(strong.alpha_):.6f}")

for tq in (0.25, 0.28, 0.32):
    save_submission(sessions, pred_strong, test_vol, tq, f"ridge_fullimpacts_strong_tq{int(tq*100):02d}.csv")

m2000 = Ridge(alpha=2000.0)
m2000.fit(Xtr, y_train)
pred2000 = np.asarray(m2000.predict(Xte), dtype=float)
for tq in (0.30, 0.25):
    save_submission(sessions, pred2000, test_vol, tq, f"ridge_fullimpacts_a2000_tq{int(tq*100):02d}.csv")
