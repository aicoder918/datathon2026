"""Deeper negative-floor continuation for full-impact strong tq30.

Outputs:
  - submissions/chatgpt/ridge_fullimpacts_strong_tq30_floor-400.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq30_floor-500.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq30_floor-700.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq30_floor-1000.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
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


def floor_tag(floor: float) -> str:
    sign = "-" if floor < 0 else ""
    return f"{sign}{abs(int(round(floor * 1000))):03d}"


def load_fullimpact_matrix() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_base, y_train, headlines_train, bars_train = load_train_base()
    impacts = fit_template_impacts_multi(headlines_train, bars_train)
    sec_impacts = fit_template_impacts_sector_multi(headlines_train, bars_train)
    train_sessions = X_base.index.to_numpy()
    X_train = X_base.join(build_event_features_multi(headlines_train, train_sessions, impacts))
    X_train = X_train.join(build_event_features_sector_multi(headlines_train, train_sessions, sec_impacts))
    X_test = load_test(impacts=impacts)
    return X_train, y_train, X_test


X_train_df, y_train_s, X_test_df = load_fullimpact_matrix()
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_df.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

strong = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
strong.fit(Xtr, y_train)
pred = np.asarray(strong.predict(Xte), dtype=float)
base = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=0.30)

for floor in (-0.40, -0.50, -0.70, -1.00):
    pos = finalize(base, short_floor=floor)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": pos,
    })
    out_path = OUT_DIR / f"ridge_fullimpacts_strong_tq30_floor{floor_tag(floor)}.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())
