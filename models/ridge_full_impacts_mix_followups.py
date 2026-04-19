"""Third-wave follow-ups around the best full-impact thresholded submission.

Outputs:
  - submissions/chatgpt/ridge_fullimpacts_mix_tq30_tq40_80_20.csv
  - submissions/chatgpt/ridge_fullimpacts_mix_tq30_a2000_80_20.csv
  - submissions/chatgpt/ridge_fullimpacts_mix_tq30_bestold_90_10.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq30_floor025.csv
  - submissions/chatgpt/ridge_fullimpacts_strong_tq30_floor035.csv
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


def load_fullimpact_matrix() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_base, y_train, headlines_train, bars_train = load_train_base()
    impacts = fit_template_impacts_multi(headlines_train, bars_train)
    sec_impacts = fit_template_impacts_sector_multi(headlines_train, bars_train)
    train_sessions = X_base.index.to_numpy()
    X_train = X_base.join(build_event_features_multi(headlines_train, train_sessions, impacts))
    X_train = X_train.join(build_event_features_sector_multi(headlines_train, train_sessions, sec_impacts))
    X_test = load_test(impacts=impacts)
    return X_train, y_train, X_test


def save_series(sessions: np.ndarray, values: pd.Series | np.ndarray, name: str) -> None:
    if isinstance(values, pd.Series):
        arr = values.reindex(sessions).to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    submission = pd.DataFrame({"session": sessions.astype(int), "target_position": arr})
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())


X_train_df, y_train_s, X_test_df = load_fullimpact_matrix()
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_df.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

strong = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
strong.fit(Xtr, y_train)
pred_strong = np.asarray(strong.predict(Xte), dtype=float)
pos_tq30 = pd.Series(
    finalize(shape_positions(pred_strong, test_vol, "thresholded_inv_vol", threshold_q=0.30)),
    index=sessions,
)
pos_tq40 = pd.read_csv(OUT_DIR / "ridge_fullimpacts_strong_tq40.csv").set_index("session")["target_position"].sort_index()
pos_a2000_tq30 = pd.read_csv(OUT_DIR / "ridge_fullimpacts_a2000_tq30.csv").set_index("session")["target_position"].sort_index()
pos_bestold = pd.read_csv(OUT_DIR / "ridge_blend_a3000_strong_w25.csv").set_index("session")["target_position"].sort_index()

mixes = {
    "ridge_fullimpacts_mix_tq30_tq40_80_20.csv": 0.80 * pos_tq30 + 0.20 * pos_tq40,
    "ridge_fullimpacts_mix_tq30_a2000_80_20.csv": 0.80 * pos_tq30 + 0.20 * pos_a2000_tq30,
    "ridge_fullimpacts_mix_tq30_bestold_90_10.csv": 0.90 * pos_tq30 + 0.10 * pos_bestold,
}
for name, series in mixes.items():
    save_series(sessions, series, name)

for floor in (0.25, 0.35):
    pos = finalize(
        shape_positions(pred_strong, test_vol, "thresholded_inv_vol", threshold_q=0.30),
        short_floor=floor,
    )
    save_series(sessions, pos, f"ridge_fullimpacts_strong_tq30_floor{int(floor*100):03d}.csv")
