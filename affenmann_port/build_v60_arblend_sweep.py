"""v60: v59_arblend_w15_amp120_sp = 2.89807 (NEW BEST). Fine-sweep around
ar-blend weight, amp, softpos params."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT / "submissions"

# Use v54 winner as the reshape base (since w15*ar was blended into BEST not v54)
# Wait — v59 used: (1-w) * best + w * ar, then amp, then sp
# where best = v54_amp110_sp_t125_m085
v54 = pd.read_csv(OUT / "v54_amp110_sp_t125_m085.csv").set_index("session")["target_position"]
ar = pd.read_csv(ROOT / "autoresearch" / "submissions" / "ar_current.csv").set_index("session")["target_position"].reindex(v54.index)

def rescale(x, s):
    mu = x.mean(); return mu + s * (x - mu)

def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    if sides in ("pos", "both"):
        m = pos > th; out[m] = m_scale * pos[m] + (1 - m_scale)
    if sides in ("neg", "both"):
        m = pos < -th; out[m] = m_scale * pos[m] - (1 - m_scale)
    return out

# Winner: w=0.15, amp=1.20, t=1.25, m=0.85
# Fine-sweep
print("=== arblend sweep ===")
count = 0
for w in [0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25]:
    blend = (1 - w) * v54 + w * ar
    for amp in [1.10, 1.15, 1.18, 1.20, 1.22, 1.25, 1.30]:
        for t, ms in [(1.25, 0.85), (1.25, 0.80), (1.25, 0.90), (1.20, 0.85), (1.30, 0.85)]:
            y = rescale(blend, amp); y = softpos(y, t, ms, "pos")
            name = f"v60_w{int(w*100):02d}_a{int(amp*100):03d}_t{int(t*100):03d}_m{int(ms*100):03d}.csv"
            pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)
            count += 1

# Also try BOTH-tails softpos (neg side clamp too)
print("=== arblend + both-tails ===")
for w in [0.12, 0.15, 0.18, 0.20]:
    blend = (1 - w) * v54 + w * ar
    for amp in [1.15, 1.20, 1.25]:
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "both")
        name = f"v60_both_w{int(w*100):02d}_a{int(amp*100):03d}.csv"
        pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Try blending ar BEFORE v54 (reverse order)
print("=== reverse: ar blended into orig first ===")
orig = pd.read_csv(OUT / "aff7_template044_ridge1000_58_42_amp140_softpos_pos_t125_m085.csv").set_index("session")["target_position"]
rq = pd.read_csv(OUT / "aff10_best_robustq01blend_95_05.csv").set_index("session")["target_position"]
rr = (rq - 0.95 * orig) / 0.05
agg = pd.read_csv("/tmp/ts_aggressive.csv").set_index("session")["target_position"].reindex(orig.index)
# 4-signal blend: orig + rr + agg + ar
for (wo, wr, wa, war) in [
    (0.85, 0.05, 0.05, 0.05),
    (0.84, 0.05, 0.05, 0.06),
    (0.82, 0.05, 0.05, 0.08),
    (0.80, 0.05, 0.05, 0.10),
    (0.78, 0.05, 0.05, 0.12),
    (0.75, 0.05, 0.05, 0.15),
]:
    blend = wo*orig + wr*rr + wa*agg + war*ar
    for amp in [1.15, 1.20, 1.25]:
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
        name = f"v60_4way_wo{int(wo*100):02d}_wr{int(wr*100):02d}_wa{int(wa*100):02d}_war{int(war*100):02d}_a{int(amp*100):03d}.csv"
        pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)
        count += 1

print(f"wrote {count} v60 candidates")

# Sanity — print winner stats
blend = 0.85 * v54 + 0.15 * ar
y = rescale(blend, 1.20); y = softpos(y, 1.25, 0.85, "pos")
print(f"winner (w15 a120 t125 m85): mean={y.mean():.4f} std={y.std():.4f} min={y.min():.4f} max={y.max():.4f}")
