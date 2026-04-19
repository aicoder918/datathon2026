"""v66: v64_cx_ar16_cx02_a122 = 2.90074 NEW BEST!
Key finding: ctxvol at 2% + higher amp (1.22) is the combo.
Sweep amp more aggressively around 1.22; also explore cx weights."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT/"submissions"
v54 = pd.read_csv(OUT/"v54_amp110_sp_t125_m085.csv").set_index("session")["target_position"]
ar = pd.read_csv(ROOT/"autoresearch/submissions/ar_current.csv").set_index("session")["target_position"].reindex(v54.index)
ctxvol = pd.read_csv(ROOT/"autoresearch/submissions/aff_ctxvol.csv").set_index("session")["target_position"].reindex(v54.index)

def rescale(x, s):
    mu = x.mean(); return mu + s * (x - mu)
def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    m = pos > th; out[m] = m_scale*pos[m] + (1 - m_scale)
    return out

count = 0
# --- A: Finer amp sweep around 1.22, with cx=2% (winning combo)
for w_ar100 in [14, 15, 16, 17, 18]:
    for w_cx100 in [1, 2, 3, 4, 5]:
        w_ar = w_ar100/100; w_cx = w_cx100/100
        w_v = 1 - w_ar - w_cx
        blend = w_v*v54 + w_ar*ar + w_cx*ctxvol
        for amp100 in [120, 121, 122, 123, 124, 125]:
            amp = amp100/100
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85)
            name = f"v66_ar{w_ar100:02d}_cx{w_cx100:02d}_a{amp100:03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- B: Even higher amp (maybe more headroom)
for w_ar100 in [15, 16, 17]:
    for w_cx100 in [2, 3, 4]:
        w_ar = w_ar100/100; w_cx = w_cx100/100
        w_v = 1 - w_ar - w_cx
        blend = w_v*v54 + w_ar*ar + w_cx*ctxvol
        for amp100 in [126, 128, 130, 132, 135]:
            amp = amp100/100
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85)
            name = f"v66H_ar{w_ar100:02d}_cx{w_cx100:02d}_a{amp100:03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- C: Softpos params around winner (cx=2, a=1.22)
w_v, w_ar, w_cx = 0.82, 0.16, 0.02
blend = w_v*v54 + w_ar*ar + w_cx*ctxvol
for t100 in [120, 125, 130]:
    for m100 in [82, 84, 85, 86, 88]:
        y = rescale(blend, 1.22); y = softpos(y, t100/100, m100/100)
        name = f"v66sp_t{t100:03d}_m{m100:03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

print(f"wrote {count} v66 candidates")
