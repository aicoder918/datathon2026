"""v65: Finer w-sweep at 0.5% resolution around w=0.16 peak.
Also test finer amp/m_scale around the winning (1.20, 0.85)."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT/"submissions"
v54 = pd.read_csv(OUT/"v54_amp110_sp_t125_m085.csv").set_index("session")["target_position"]
ar = pd.read_csv(ROOT/"autoresearch/submissions/ar_current.csv").set_index("session")["target_position"].reindex(v54.index)

def rescale(x, s):
    mu = x.mean(); return mu + s * (x - mu)
def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    m = pos > th; out[m] = m_scale*pos[m] + (1 - m_scale)
    return out

count = 0
# --- A: w in 0.5% steps from 0.145 to 0.185
for w1000 in [145, 150, 155, 158, 160, 162, 165, 168, 170, 175, 180, 185]:
    w = w1000/1000
    blend = (1-w)*v54 + w*ar
    for amp100 in [118, 119, 120, 121, 122]:
        amp = amp100/100
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85)
        name = f"v65_w{w1000:03d}_a{amp100:03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- B: Peak (w=0.16, a=1.20) with finer softpos params
w = 0.16; blend = (1-w)*v54 + w*ar
for t100 in [120, 122, 125, 128, 130]:
    for m100 in [82, 84, 85, 86, 88, 90]:
        y = rescale(blend, 1.20); y = softpos(y, t100/100, m100/100)
        name = f"v65_sp_t{t100:03d}_m{m100:03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- C: Two-pass softpos (apply twice with different params)
for (t1, m1, t2, m2) in [(1.25, 0.85, 1.50, 0.90),(1.00, 0.90, 1.50, 0.85),(1.25, 0.90, 1.00, 0.85),(1.50, 0.85, 1.00, 0.90)]:
    y = rescale(blend, 1.20); y = softpos(y, t1, m1); y = softpos(y, t2, m2)
    name = f"v65_2sp_{int(t1*100):03d}{int(m1*100):03d}_{int(t2*100):03d}{int(m2*100):03d}.csv"
    pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
    count += 1

print(f"wrote {count} v65 candidates")
