"""v55: v54_amp110_sp_t125_m085 = 2.88951 — huge jump. Fine-sweep amp scale +
softpos params around the winner to pinpoint the peak."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT / "submissions"

# Base = v51_tsagg06 (the 3-signal blend, before reshape)
base = pd.read_csv(OUT / "v51_tsagg06.csv").set_index("session")["target_position"]


def rescale(x, s):
    mu = x.mean()
    return mu + s * (x - mu)


def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    if sides in ("pos", "both"):
        mask = pos > th
        out[mask] = m_scale * pos[mask] + (1 - m_scale)
    if sides in ("neg", "both"):
        mask = pos < -th
        out[mask] = m_scale * pos[mask] - (1 - m_scale)
    return out


# Winner: amp=1.10, t=1.25, m=0.85. Fine sweep near it.
print("=== fine sweep around amp110_t125_m085 ===")
candidates = []
for s in [1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.15]:
    for t in [1.20, 1.225, 1.25, 1.275, 1.30]:
        for m in [0.80, 0.825, 0.85, 0.875, 0.90]:
            y = rescale(base, s)
            y = softpos(y, t, m, "pos")
            name = f"v55_amp{int(s*1000):04d}_t{int(t*1000):04d}_m{int(m*1000):03d}.csv"
            pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)
            candidates.append(name)

# Also try BOTH tails (pos + neg) — the blend has some negative values
print("=== amp + softpos BOTH ===")
for s in [1.08, 1.10, 1.12]:
    for t in [1.20, 1.25, 1.30]:
        for m in [0.80, 0.85, 0.90]:
            y = rescale(base, s)
            y = softpos(y, t, m, "both")
            name = f"v55_both_amp{int(s*1000):04d}_t{int(t*1000):04d}_m{int(m*1000):03d}.csv"
            pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Bigger amp scales (maybe sweet spot is >1.15)
print("=== higher amp scales ===")
for s in [1.17, 1.20, 1.25]:
    for t, m in [(1.25, 0.85), (1.30, 0.85), (1.25, 0.80)]:
        y = rescale(base, s)
        y = softpos(y, t, m, "pos")
        name = f"v55_ampHi{int(s*100)}_t{int(t*100)}_m{int(m*100):03d}.csv"
        pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

print(f"candidates: {len(candidates)}")
print("done")
