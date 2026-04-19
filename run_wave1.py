"""Wave 1 experiments on top of 2.90305 champion.

Champion: submissions/submission_best_plus_ridge_top10_935_065.csv
= 0.935 * BASE + 0.065 * ridge_top10
where BASE = submission_best_plus_seed20_900_100_amp105_seed5decay_0910_0090_amp105_x103.csv (LB 2.90190)

Hypothesis: other orthogonal ridge/pls branches may match or beat ridge_top10 at same weight,
or complement it as a 3rd branch on the champion.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
SUB = ROOT / "submissions"

BASE = SUB / "submission_best_plus_seed20_900_100_amp105_seed5decay_0910_0090_amp105_x103.csv"
CHAMP = SUB / "submission_best_plus_ridge_top10_935_065.csv"
OUT_DIR = SUB
OUT_DIR.mkdir(exist_ok=True)

def load(p):
    return pd.read_csv(p).sort_values("session").reset_index(drop=True)

def write(df, name, desc):
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"{name:80s}  mean={df.target_position.mean():.4f} std={df.target_position.std():.4f} min={df.target_position.min():.3f} max={df.target_position.max():.3f}  [{desc}]")
    return path

def blend(left, right, w, name, desc):
    L = load(left); R = load(right)
    assert L["session"].equals(R["session"])
    out = L.copy()
    out["target_position"] = w * L["target_position"].values + (1-w) * R["target_position"].values
    return write(out, name, desc)

base = BASE
champ = CHAMP

# --- Part A: substitute the second branch at 0.935 / 0.065 (same weight as champion's winning split) ---
substitutes = [
    ("chatgpt/ridge_all_strong.csv",          "sub_base_plus_ridgeallstr_935_065.csv"),
    ("chatgpt/ridge_alpha_50000.csv",         "sub_base_plus_ridgea50k_935_065.csv"),
    ("chatgpt/ridge_fullimpacts_strong_tq30.csv", "sub_base_plus_ridgefitq30_935_065.csv"),
    ("chatgpt/pls_c5.csv",                    "sub_base_plus_plsc5_935_065.csv"),
    ("chatgpt/pls_c3.csv",                    "sub_base_plus_plsc3_935_065.csv"),
]
for src, name in substitutes:
    blend(base, SUB / src, 0.935, name, f"0.935*base + 0.065*{src}")

# --- Part B: add a 3rd orthogonal branch on top of champion (99/1 and 98/2) ---
thirds = [
    ("chatgpt/ridge_alpha_50000.csv",  "champ_plus_ridgea50k"),
    ("chatgpt/pls_c5.csv",             "champ_plus_plsc5"),
    ("chatgpt/catboost_bars_bootstrap30.csv", "champ_plus_cb30"),
]
for src, stub in thirds:
    for w in (0.99, 0.98):
        wd = f"{int(w*1000):03d}"
        ow = f"{int((1-w)*1000):03d}"
        name = f"{stub}_{wd}_{ow}.csv"
        blend(champ, SUB / src, w, name, f"{w}*champ + {1-w}*{src}")
