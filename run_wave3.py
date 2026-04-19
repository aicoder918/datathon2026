"""Wave 3: refine weights near public-LB winners (2026-04-19 snapshot).

Evidence from `kaggle competitions submissions`:
  ~2.90306 cluster: sub_base_rt10_ra50 at (935,55,10), (935,60,5), (930,60,10), (930,55,15)
  ~2.90305: champ_plus_ridgea50k_995_005
  Multi-aux ridge937 (octa→quad) trailed at 2.892–2.901 — not worth more API slots.

This script writes a small simplex grid around the 3-branch winners + a finer
champ+ridge_alpha_50000 strip. Review locally (corr vs anchors) then submit
only a handful.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SUB = ROOT / "submissions"
BASE = SUB / "submission_best_plus_seed20_900_100_amp105_seed5decay_0910_0090_amp105_x103.csv"
CHAMP = SUB / "submission_best_plus_ridge_top10_935_065.csv"
RT10 = SUB / "chatgpt" / "ridge_top10.csv"
RA50 = SUB / "chatgpt" / "ridge_alpha_50000.csv"


def load(p: Path) -> pd.DataFrame:
    return pd.read_csv(p).sort_values("session").reset_index(drop=True)


def mix(frames_weights: list[tuple[Path, float]], name: str, desc: str) -> None:
    sessions = None
    out = None
    for f, w in frames_weights:
        d = load(f)
        if sessions is None:
            sessions = d["session"]
            out = d.copy()
            out["target_position"] = 0.0
        assert d["session"].equals(sessions)
        out["target_position"] = out["target_position"].values + w * d["target_position"].values
    p = SUB / name
    out.to_csv(p, index=False)
    x = out["target_position"]
    print(f"{name:55s}  mean={x.mean():.4f} std={x.std():.4f}  [{desc}]")


def blend(left: Path, right: Path, w: float, name: str, desc: str) -> None:
    L = load(left)
    R = load(right)
    assert L["session"].equals(R["session"])
    out = L.copy()
    out["target_position"] = w * L["target_position"].values + (1 - w) * R["target_position"].values
    out.to_csv(SUB / name, index=False)
    x = out["target_position"]
    print(f"{name:55s}  mean={x.mean():.4f} std={x.std():.4f}  [{desc}]")


# --- A) 3-branch tight mesh: wb + wrt + wra = 1, focus near LB-tied vertices ---
seen: set[tuple[int, int, int]] = set()
for wb in [0.931, 0.932, 0.933, 0.934, 0.935, 0.936, 0.937, 0.938]:
    for wrt in [0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.060, 0.061, 0.062]:
        wra = 1.0 - wb - wrt
        if not (0.004 <= wra <= 0.014):
            continue
        tri = (
            int(round(wb * 1000)),
            int(round(wrt * 1000)),
            int(round(wra * 1000)),
        )
        if tri in seen:
            continue
        seen.add(tri)
        name = f"sub_w3_rt10_ra50_{tri[0]:03d}_{tri[1]:03d}_{tri[2]:03d}.csv"
        mix(
            [(BASE, wb), (RT10, wrt), (RA50, wra)],
            name,
            f"{wb:.3f}*base+{wrt:.3f}*rt10+{wra:.3f}*ra50",
        )

# --- B) champ + ridge_alpha_50000 near 99.5/0.5 winner ---
for w in [0.993, 0.994, 0.995, 0.996, 0.997, 0.998]:
    wd = f"{int(round(w * 1000)):03d}"
    ow = f"{int(round((1 - w) * 1000)):03d}"
    blend(CHAMP, RA50, w, f"sub_w3_champ_ra50_{wd}_{ow}.csv", f"{w}*champ + {1-w}*ra50")
