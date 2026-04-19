"""Focused follow-up search around the best full-impact ridge models.

Outputs:
  - submissions/chatgpt/ridge_fullimpacts_blend_a3000_strong_w50.csv
  - submissions/chatgpt/ridge_fullimpacts_blend_a3000_strong_w75.csv
  - submissions/chatgpt/ridge_fullimpacts_mix_strong_best_70_30.csv
  - submissions/chatgpt/ridge_fullimpacts_postrim_top3.csv
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
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_df.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

m3000 = Ridge(alpha=3000.0)
m3000.fit(Xtr, y_train)
pred3000 = np.asarray(m3000.predict(Xte), dtype=float)

mstrong = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
mstrong.fit(Xtr, y_train)
predstrong = np.asarray(mstrong.predict(Xte), dtype=float)
print(f"fullimpact strong alpha={float(mstrong.alpha_):.6f}")

raw_variants = {
    "ridge_fullimpacts_blend_a3000_strong_w50.csv": 0.50 * pred3000 + 0.50 * predstrong,
    "ridge_fullimpacts_blend_a3000_strong_w75.csv": 0.25 * pred3000 + 0.75 * predstrong,
}

for name, pred in raw_variants.items():
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({"session": sessions.astype(int), "target_position": pos})
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())

pos_fullstrong = pd.read_csv(OUT_DIR / "ridge_fullimpacts_strong.csv").set_index("session")["target_position"].sort_index()
pos_best = pd.read_csv(OUT_DIR / "ridge_blend_a3000_strong_w25.csv").set_index("session")["target_position"].sort_index()
pos_fulla3000 = pd.read_csv(OUT_DIR / "ridge_fullimpacts_a3000.csv").set_index("session")["target_position"].sort_index()

position_variants = {
    "ridge_fullimpacts_mix_strong_best_70_30.csv": 0.70 * pos_fullstrong + 0.30 * pos_best,
    "ridge_fullimpacts_postrim_top3.csv": pd.concat(
        [pos_fullstrong, pos_fulla3000, pos_best], axis=1
    ).apply(lambda row: np.sort(row.to_numpy(dtype=float))[1], axis=1),
}

for name, series in position_variants.items():
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": series.reindex(sessions).to_numpy(dtype=float),
    })
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())
