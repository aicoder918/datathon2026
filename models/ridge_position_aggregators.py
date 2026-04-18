"""Robust position-space aggregators over good ridge submissions.

Outputs:
  - submissions/chatgpt/ridge_posblend_a3000_a10000_55_45.csv
  - submissions/chatgpt/ridge_posmedian_a3000_a10000_a20000.csv
  - submissions/chatgpt/ridge_postrim_top4.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "submissions" / "chatgpt"


def load_positions(name: str) -> pd.Series:
    df = pd.read_csv(SUB_DIR / name)
    return df.set_index("session")["target_position"].sort_index()


a3000 = load_positions("ridge_alpha_3000.csv")
a10000 = load_positions("ridge_alpha_10000.csv")
a20000 = load_positions("ridge_alpha_20000.csv")
strong = load_positions("ridge_all_strong.csv")
blend25 = load_positions("ridge_blend_a3000_strong_w25.csv")
blend50 = load_positions("ridge_blend_a3000_strong_w50.csv")
sessions = a3000.index

variants = {
    "ridge_posblend_a3000_a10000_55_45.csv": 0.55 * a3000 + 0.45 * a10000,
    "ridge_posmedian_a3000_a10000_a20000.csv": pd.concat(
        [a3000, a10000, a20000], axis=1
    ).median(axis=1),
    "ridge_postrim_top4.csv": pd.concat(
        [strong, a3000, blend25, blend50], axis=1
    ).apply(lambda row: np.sort(row.to_numpy(dtype=float))[1:3].mean(), axis=1),
}

for name, series in variants.items():
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": series.to_numpy(dtype=float),
    })
    out_path = SUB_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())
