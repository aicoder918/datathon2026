"""v57: Expand blend base to 4-5 diverse signals (rr, agg, lgbm, v4), then reshape.
Hypothesis: diverse blend → lower std → amp restores → more tail info per token."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT / "submissions"

orig = pd.read_csv(OUT / "aff7_template044_ridge1000_58_42_amp140_softpos_pos_t125_m085.csv").set_index("session")["target_position"]
rq = pd.read_csv(OUT / "aff10_best_robustq01blend_95_05.csv").set_index("session")["target_position"]
rr = (rq - 0.95 * orig) / 0.05
agg = pd.read_csv("/tmp/ts_aggressive.csv").set_index("session")["target_position"].reindex(orig.index)
v4 = pd.read_csv("/tmp/ts_v4.csv").set_index("session")["target_position"].reindex(orig.index)
lgbm = pd.read_csv("/tmp/ts_lgbm_featuregen.csv").set_index("session")["target_position"].reindex(orig.index)
tpl = pd.read_csv("/tmp/ts_template.csv").set_index("session")["target_position"].reindex(orig.index)

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

# v54 winner = 0.89 orig + 0.05 rr + 0.06 agg, amp1.10, sp t1.25 m0.85
# v57 explores larger diverse-signal pools before reshape

blends = {
    # Baseline (match v54)
    "b4a6": (0.89, 0.05, 0.06, 0.00, 0.00),  # (orig, rr, agg, v4, lgbm)
    # Add v4 (corr 0.35 — diverse from agg corr 0.37)
    "b4a4_v42": (0.88, 0.04, 0.04, 0.02, 0.02),
    "b4a4_v43": (0.87, 0.04, 0.04, 0.03, 0.02),
    "b4a3_v44": (0.86, 0.04, 0.03, 0.04, 0.03),
    "b6a4l3":   (0.87, 0.04, 0.04, 0.02, 0.03),
    # Push more rr
    "rr07a5":  (0.83, 0.07, 0.05, 0.03, 0.02),
    "rr10a5":  (0.80, 0.10, 0.05, 0.03, 0.02),
    # Heavy diversity
    "heavy":   (0.82, 0.05, 0.04, 0.04, 0.05),
    "rrheavy": (0.80, 0.06, 0.04, 0.04, 0.06),
    # Concentrate on orig
    "orig91":  (0.91, 0.03, 0.03, 0.02, 0.01),
}

# For each blend, sweep amp {1.08, 1.10, 1.12, 1.15} and softpos standard
amp_grid = [1.08, 1.10, 1.12, 1.15, 1.20]
sp_combos = [(1.25, 0.85), (1.25, 0.80), (1.20, 0.85), (1.30, 0.85)]

count = 0
for bname, (wo, wr, wa, wv, wl) in blends.items():
    blend = wo*orig + wr*rr + wa*agg + wv*v4 + wl*lgbm
    for amp in amp_grid:
        for t, m in sp_combos:
            y = rescale(blend, amp)
            y = softpos(y, t, m, "pos")
            name = f"v57_{bname}_a{int(amp*100):03d}_t{int(t*100):03d}_m{int(m*100):03d}.csv"
            pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)
            count += 1

print(f"wrote {count} v57 candidates")

# Sanity — print top blends pre-reshape stats
for bname, (wo, wr, wa, wv, wl) in blends.items():
    blend = wo*orig + wr*rr + wa*agg + wv*v4 + wl*lgbm
    print(f"{bname:12} mean={blend.mean():.4f} std={blend.std():.4f} corr_orig={np.corrcoef(blend, orig)[0,1]:.3f}")
