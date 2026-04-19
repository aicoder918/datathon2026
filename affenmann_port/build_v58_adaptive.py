"""v58: Adaptive (per-sample) amplification — high-confidence sessions get more amp,
low-confidence less. Hypothesis: global amp = 1.10 under-amps the best sessions and
over-amps the noisy ones. Use position magnitude as a confidence proxy."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT / "submissions"

best = pd.read_csv(ROOT / "best" / "best.csv").set_index("session")["target_position"]
orig = pd.read_csv(OUT / "aff7_template044_ridge1000_58_42_amp140_softpos_pos_t125_m085.csv").set_index("session")["target_position"]
rq = pd.read_csv(OUT / "aff10_best_robustq01blend_95_05.csv").set_index("session")["target_position"]
rr = (rq - 0.95 * orig) / 0.05
agg = pd.read_csv("/tmp/ts_aggressive.csv").set_index("session")["target_position"].reindex(orig.index)

# Base blend = v51_tsagg06 (the blend used as input to v54)
base_blend = 0.89 * orig + 0.05 * rr + 0.06 * agg

# Adaptive amp: scale s per-session based on |pos - mean|
# Sessions far from mean (extreme positions) get bigger amp.
# Near-mean (low confidence) stay calm.
mu = base_blend.mean()
dev = (base_blend - mu).abs()
# Normalize dev to [0, 1]; rank-based for robustness
rank = dev.rank(pct=True)

def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    if sides in ("pos", "both"):
        mask = pos > th
        out[mask] = m_scale * pos[mask] + (1 - m_scale)
    if sides in ("neg", "both"):
        mask = pos < -th
        out[mask] = m_scale * pos[mask] - (1 - m_scale)
    return out

# Strategy 1: piecewise amp — high-conf gets amp_hi, low-conf gets amp_lo
for amp_lo, amp_hi in [(1.00, 1.15), (1.00, 1.20), (1.05, 1.20), (0.95, 1.25), (1.00, 1.25)]:
    s = amp_lo + rank * (amp_hi - amp_lo)
    y = mu + s * (base_blend - mu)
    y = softpos(y, 1.25, 0.85, "pos")
    name = f"v58_adapt_lo{int(amp_lo*100)}_hi{int(amp_hi*100)}.csv"
    pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Strategy 2: per-sample linear blend of (base_blend, v54_best)
# best is v54 at amp1.10. Mix with base_blend for calibration
for w in [0.3, 0.5, 0.7, 0.85]:
    y = w * best + (1 - w) * base_blend
    name = f"v58_mix_best{int(w*100)}_blend{int((1-w)*100)}.csv"
    pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Strategy 3: extremely tight around v54 best (overfit-check)
# amp 1.09, 1.11 with varying t
for amp in [1.09, 1.105, 1.11]:
    for t in [1.22, 1.25, 1.28]:
        for m in [0.82, 0.85, 0.88]:
            y = mu + amp * (base_blend - mu)
            y = softpos(y, t, m, "pos")
            name = f"v58_tight_a{int(amp*1000):04d}_t{int(t*1000):04d}_m{int(m*1000):03d}.csv"
            pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Strategy 4: double reshape — amp then softpos then amp again (small, 1.02)
for amp1, amp2 in [(1.10, 1.02), (1.08, 1.05), (1.12, 0.98)]:
    y = mu + amp1 * (base_blend - mu)
    y = softpos(y, 1.25, 0.85, "pos")
    mu2 = y.mean()
    y = mu2 + amp2 * (y - mu2)
    name = f"v58_double_a1{int(amp1*100)}_a2{int(amp2*100)}.csv"
    pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

print("v58 candidates built")
