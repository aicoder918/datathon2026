"""Wave 2: micro-sweeps, 3-branch mixes, amplitude on champion."""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).parent
SUB = ROOT / "submissions"
BASE = SUB / "submission_best_plus_seed20_900_100_amp105_seed5decay_0910_0090_amp105_x103.csv"
CHAMP = SUB / "submission_best_plus_ridge_top10_935_065.csv"
RT10 = SUB / "chatgpt" / "ridge_top10.csv"
RA50 = SUB / "chatgpt" / "ridge_alpha_50000.csv"

def load(p): return pd.read_csv(p).sort_values("session").reset_index(drop=True)

def write(df, name, desc):
    p = SUB / name
    df.to_csv(p, index=False)
    d = df.target_position
    print(f"{name:60s}  mean={d.mean():.4f} std={d.std():.4f} min={d.min():.3f} max={d.max():.3f}  [{desc}]")

def mix(frames_weights, name, desc):
    sessions = None; acc = 0.0
    for f, w in frames_weights:
        d = load(f)
        if sessions is None: sessions = d["session"]; out = d.copy(); out["target_position"] = 0.0
        acc_arr = w * d["target_position"].values
        out["target_position"] = out["target_position"].values + acc_arr
    write(out, name, desc)

def amp(f, a, name, desc):
    d = load(f)
    x = d["target_position"].values
    out = d.copy()
    out["target_position"] = 1.0 + a * (x - 1.0)
    write(out, name, desc)

# A) fine ridge_top10 sweep around champ
for w in [0.905, 0.910, 0.915, 0.920, 0.925, 0.928, 0.932, 0.948, 0.960, 0.965]:
    ws = f"{int(round(w*1000)):03d}"; rs = f"{int(round((1-w)*1000)):03d}"
    mix([(BASE, w), (RT10, 1-w)], f"sub_base_plus_ridge_top10_{ws}_{rs}.csv",
        f"{w}*base + {1-w}*ridge_top10")

# B) 3-branch: base + ridge_top10 + ridge_alpha_50000 (splitting the ridge allotment)
for (wb, wrt, wra) in [(0.935,0.055,0.010), (0.935,0.060,0.005), (0.930,0.060,0.010),
                       (0.930,0.055,0.015), (0.920,0.065,0.015), (0.925,0.065,0.010)]:
    name = f"sub_base_rt10_ra50_{int(wb*1000):03d}_{int(round(wrt*1000)):03d}_{int(round(wra*1000)):03d}.csv"
    mix([(BASE, wb), (RT10, wrt), (RA50, wra)], name, f"{wb}b+{wrt}rt10+{wra}ra50")

# C) amplitude on champion
for a in [1.005, 1.010, 1.015, 1.020, 0.995, 0.990]:
    ta = f"{int(round(a*1000)):04d}"
    amp(CHAMP, a, f"champ_amp{ta}.csv", f"1+{a}*(x-1) on champ")
