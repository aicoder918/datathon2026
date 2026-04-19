"""v61: multi-signal blend. Use orthogonal diversity (ar, agg, v4)
on top of v54 (which is already orig+rr+tsagg reshape). Peak at w_ar=0.15.

Strategy: replace part of ar's contribution with orthogonal (agg, v4) mix
to get MORE diverse info without overweighting any single noisy signal."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT/"submissions"

v54 = pd.read_csv(OUT/"v54_amp110_sp_t125_m085.csv").set_index("session")["target_position"]
ar = pd.read_csv(ROOT/"autoresearch/submissions/ar_current.csv").set_index("session")["target_position"].reindex(v54.index)
agg = pd.read_csv("/tmp/ts_aggressive.csv").set_index("session")["target_position"].reindex(v54.index)
v4 = pd.read_csv("/tmp/ts_v4.csv").set_index("session")["target_position"].reindex(v54.index)
tpl = pd.read_csv("/tmp/ts_template.csv").set_index("session")["target_position"].reindex(v54.index)

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
# --- Pattern A: 3-way (v54 + ar + agg). agg orthogonal to ar (-0.016).
# Total "diverse weight" = w_ar + w_agg, keep around 0.15 (winner).
for w_ar in [0.08, 0.10, 0.12, 0.15]:
    for w_agg in [0.03, 0.05, 0.07, 0.10]:
        w_v54 = 1 - w_ar - w_agg
        if w_v54 < 0.75: continue
        blend = w_v54*v54 + w_ar*ar + w_agg*agg
        for amp in [1.18, 1.20, 1.22, 1.25]:
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
            name = f"v61_3way_v{int(w_v54*100):02d}_ar{int(w_ar*100):02d}_ag{int(w_agg*100):02d}_a{int(amp*100):03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- Pattern B: 3-way (v54 + ar + v4). v4 at 0.343 corr — similar to agg but different signal.
for w_ar in [0.10, 0.12, 0.15]:
    for w_v4 in [0.03, 0.05, 0.07]:
        w_v54 = 1 - w_ar - w_v4
        blend = w_v54*v54 + w_ar*ar + w_v4*v4
        for amp in [1.18, 1.20, 1.22]:
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
            name = f"v61_3wayV4_v{int(w_v54*100):02d}_ar{int(w_ar*100):02d}_v4{int(w_v4*100):02d}_a{int(amp*100):03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

# --- Pattern C: 4-way (v54 + ar + agg + v4). Most diverse.
for (w_ar, w_agg, w_v4) in [(0.08,0.03,0.03),(0.10,0.03,0.03),(0.10,0.05,0.03),(0.08,0.05,0.05),(0.10,0.04,0.04),(0.12,0.03,0.03)]:
    w_v54 = 1 - w_ar - w_agg - w_v4
    blend = w_v54*v54 + w_ar*ar + w_agg*agg + w_v4*v4
    for amp in [1.18, 1.20, 1.22, 1.25]:
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
        name = f"v61_4w_ar{int(w_ar*100):02d}_ag{int(w_agg*100):02d}_v4{int(w_v4*100):02d}_a{int(amp*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- Pattern D: (agg+v4)/2 as single orthogonal blend (they have corr -0.306 — cancels noise)
orth = (agg + v4) / 2  # expected corr to v54 ~0.35
print(f"orth stats: corr(v54)={v54.corr(orth):.3f} corr(ar)={ar.corr(orth):.3f}")
for w_ar in [0.10, 0.12, 0.15]:
    for w_orth in [0.03, 0.05, 0.07, 0.10]:
        w_v54 = 1 - w_ar - w_orth
        blend = w_v54*v54 + w_ar*ar + w_orth*orth
        for amp in [1.18, 1.20, 1.22]:
            y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
            name = f"v61_orth_ar{int(w_ar*100):02d}_o{int(w_orth*100):02d}_a{int(amp*100):03d}.csv"
            pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
            count += 1

print(f"wrote {count} v61 candidates")
# Print some sanity checks
for key_name in ["v61_3way_v82_ar10_ag05_a120", "v61_4w_ar10_ag03_v403_a120", "v61_orth_ar12_o05_a120"]:
    f = OUT/f"{key_name}.csv"
    if f.exists():
        x = pd.read_csv(f)["target_position"]
        print(f"{key_name}: mean={x.mean():.4f} std={x.std():.4f} min={x.min():.3f} max={x.max():.3f}")
