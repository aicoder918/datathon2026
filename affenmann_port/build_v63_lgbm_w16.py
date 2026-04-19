"""v63: Peak shifted to w=0.16. Fine-sweep + LGBM signal blend.
LGBM signal: corr(best)=0.143 (very diverse!), low-sharpe so use small weight."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT/"submissions"

v54 = pd.read_csv(OUT/"v54_amp110_sp_t125_m085.csv").set_index("session")["target_position"]
ar = pd.read_csv(ROOT/"autoresearch/submissions/ar_current.csv").set_index("session")["target_position"].reindex(v54.index)
lgbm = pd.read_csv(ROOT/"autoresearch/submissions/lgbm_sharpe_300.csv").set_index("session")["target_position"].reindex(v54.index)

def rescale(x, s):
    mu = x.mean(); return mu + s * (x - mu)
def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    if sides in ("pos","both"):
        m = pos > th; out[m] = m_scale*pos[m] + (1 - m_scale)
    if sides in ("neg","both"):
        m = pos < -th; out[m] = m_scale*pos[m] - (1 - m_scale)
    return out

count = 0

# --- A: fine w sweep around NEW peak w=0.16 (replacement for v60)
for w100 in range(14, 20):
    w = w100/100
    blend = (1-w)*v54 + w*ar
    for amp in [1.18, 1.20, 1.22]:
        for tms in [(1.25,0.85),(1.25,0.80),(1.25,0.90),(1.20,0.85),(1.30,0.85)]:
            t, ms = tms
            y = rescale(blend, amp); y = softpos(y, t, ms, "pos")
            name = f"v63_w{w100:02d}_a{int(amp*100):03d}_t{int(t*100):03d}_m{int(ms*100):03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- B: LGBM blend (small weight since in-sample sharpe low)
for w_ar in [0.14, 0.15, 0.16, 0.17]:
    for w_lg in [0.02, 0.03, 0.05, 0.07, 0.10]:
        w_v54 = 1 - w_ar - w_lg
        blend = w_v54*v54 + w_ar*ar + w_lg*lgbm
        for amp in [1.18, 1.20, 1.22]:
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
            name = f"v63_lg_ar{int(w_ar*100):02d}_lg{int(w_lg*100):02d}_a{int(amp*100):03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- C: LGBM only as diverse signal (no ar, to test if LGBM can REPLACE ar)
for w_lg in [0.05, 0.10, 0.15, 0.20]:
    blend = (1-w_lg)*v54 + w_lg*lgbm
    for amp in [1.15, 1.20, 1.25]:
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
        name = f"v63_lgonly_w{int(w_lg*100):02d}_a{int(amp*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

print(f"wrote {count} v63 candidates")
# Sanity
for name in ["v63_w16_a120_t125_m085", "v63_lg_ar16_lg05_a120", "v63_lgonly_w15_a120"]:
    f = OUT/f"{name}.csv"
    if f.exists():
        x = pd.read_csv(f)["target_position"]
        print(f"{name}: mean={x.mean():.4f} std={x.std():.4f} min={x.min():.3f} max={x.max():.3f}")
