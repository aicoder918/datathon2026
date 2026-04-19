"""Local sweep around the current best full-impact ridge model.

Outputs:
  - submissions/chatgpt/ridge_fullimpacts_a2000.csv
  - submissions/chatgpt/ridge_fullimpacts_a2500.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq30.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq40.csv
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


def save_submission(
    sessions: np.ndarray,
    pred: np.ndarray,
    test_vol: np.ndarray,
    threshold_q: float,
    name: str,
) -> None:
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=threshold_q)
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

for alpha in (2000.0, 2500.0):
    model = Ridge(alpha=alpha)
    model.fit(Xtr, y_train)
    pred = np.asarray(model.predict(Xte), dtype=float)
    save_submission(sessions, pred, test_vol, 0.35, f"ridge_fullimpacts_a{int(alpha)}.csv")

strong = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
strong.fit(Xtr, y_train)
pred_strong = np.asarray(strong.predict(Xte), dtype=float)
print(f"strong alpha={float(strong.alpha_):.6f}")
for tq in (0.30, 0.40):
    save_submission(sessions, pred_strong, test_vol, tq, f"ridge_fullimpacts_strong_tq{int(tq*100):02d}.csv")
