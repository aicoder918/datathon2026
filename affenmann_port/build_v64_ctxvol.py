"""v64: Affenmann's ctxvol_ols_ctx1 is ULTRA-diverse!
corr(best)=0.170, corr(ar)=0.204, corr(lgbm)=-0.003.
std=1.42 (very wide). Use at SMALL weight to avoid washing out signal."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT/"submissions"

v54 = pd.read_csv(OUT/"v54_amp110_sp_t125_m085.csv").set_index("session")["target_position"]
ar = pd.read_csv(ROOT/"autoresearch/submissions/ar_current.csv").set_index("session")["target_position"].reindex(v54.index)
ctxvol = pd.read_csv(ROOT/"autoresearch/submissions/aff_ctxvol.csv").set_index("session")["target_position"].reindex(v54.index)
tpl = pd.read_csv(ROOT/"autoresearch/submissions/aff_tplzsum.csv").set_index("session")["target_position"].reindex(v54.index)
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

# Normalize ctxvol to have same mean/std as ar (so blend weights mean the same thing)
# BUT — don't do this yet, test both normalized and raw. Raw first.
count = 0

# --- A: v54 + ar (peak) + ctxvol (tiny weight, raw)
for w_ar in [0.14, 0.16, 0.17]:
    for w_cx in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        w_v54 = 1 - w_ar - w_cx
        blend = w_v54*v54 + w_ar*ar + w_cx*ctxvol
        for amp in [1.18, 1.20, 1.22]:
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
            name = f"v64_cx_ar{int(w_ar*100):02d}_cx{int(w_cx*100):02d}_a{int(amp*100):03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- B: ctxvol normalized (match ar's scale)
# ar: std ~ 0.314. ctxvol: std ~ 1.42. Scale ctxvol down.
ctxvol_n = ctxvol.mean() + (ar.std()/ctxvol.std()) * (ctxvol - ctxvol.mean())
for w_ar in [0.14, 0.16]:
    for w_cxn in [0.03, 0.05, 0.07, 0.10, 0.12]:
        w_v54 = 1 - w_ar - w_cxn
        blend = w_v54*v54 + w_ar*ar + w_cxn*ctxvol_n
        for amp in [1.18, 1.20, 1.22]:
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
            name = f"v64_cxN_ar{int(w_ar*100):02d}_cx{int(w_cxn*100):02d}_a{int(amp*100):03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- C: 4-way with ctxvol (ultra-diverse) + lgbm (also diverse, corr -0.003 with ctxvol!)
for (w_ar, w_cx, w_lg) in [(0.14, 0.03, 0.03),(0.14, 0.03, 0.05),(0.15, 0.02, 0.03),(0.16, 0.02, 0.03),(0.13, 0.04, 0.03),(0.15, 0.03, 0.05)]:
    w_v54 = 1 - w_ar - w_cx - w_lg
    blend = w_v54*v54 + w_ar*ar + w_cx*ctxvol + w_lg*lgbm
    for amp in [1.18, 1.20, 1.22]:
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
        name = f"v64_4d_ar{int(w_ar*100):02d}_cx{int(w_cx*100):02d}_lg{int(w_lg*100):02d}_a{int(amp*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- D: ctxvol only (no ar) at tiny weight to test raw value
for w_cx in [0.02, 0.03, 0.05, 0.07]:
    blend = (1-w_cx)*v54 + w_cx*ctxvol
    for amp in [1.18, 1.20, 1.22]:
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
        name = f"v64_cxonly_w{int(w_cx*100):02d}_a{int(amp*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

print(f"wrote {count} v64 candidates")
for name in ["v64_cx_ar16_cx02_a120", "v64_cxN_ar16_cx07_a120", "v64_4d_ar14_cx03_lg03_a120", "v64_cxonly_w03_a120"]:
    f = OUT/f"{name}.csv"
    if f.exists():
        x = pd.read_csv(f)["target_position"]
        print(f"{name}: mean={x.mean():.4f} std={x.std():.4f} min={x.min():.3f} max={x.max():.3f}")
