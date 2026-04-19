"""v59: Stop incremental reshape. Try genuinely different ensembles and transforms.
Current best v54 has corr 0.999 with every nearby tweak — no info gain from sweeps.
Need structurally different candidates."""
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
v4 = pd.read_csv("/tmp/ts_v4.csv").set_index("session")["target_position"].reindex(orig.index)
lgbm = pd.read_csv("/tmp/ts_lgbm_featuregen.csv").set_index("session")["target_position"].reindex(orig.index)
tpl = pd.read_csv("/tmp/ts_template.csv").set_index("session")["target_position"].reindex(orig.index)
ar = pd.read_csv(ROOT / "autoresearch" / "submissions" / "ar_current.csv").set_index("session")["target_position"].reindex(orig.index)

def rescale(x, s):
    mu = x.mean(); return mu + s * (x - mu)

def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    if sides in ("pos", "both"):
        m = pos > th; out[m] = m_scale * pos[m] + (1 - m_scale)
    if sides in ("neg", "both"):
        m = pos < -th; out[m] = m_scale * pos[m] - (1 - m_scale)
    return out

# Strategy A: Much heavier amps
print("=== heavy amp sweep ===")
b = 0.89*orig + 0.05*rr + 0.06*agg  # same base
for amp in [1.25, 1.30, 1.40, 1.50]:
    for t in [1.25, 1.40, 1.60]:
        for m in [0.75, 0.85, 0.95]:
            y = rescale(b, amp); y = softpos(y, t, m, "pos")
            name = f"v59_heavyamp_a{int(amp*100)}_t{int(t*100)}_m{int(m*100):03d}.csv"
            pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Strategy B: Heavy autoresearch weight (corr 0.607 — really diverse)
print("=== ar blend ===")
for w_ar in [0.05, 0.10, 0.15, 0.20, 0.30]:
    y = (1 - w_ar) * best + w_ar * ar
    name = f"v59_arblend_w{int(w_ar*100):02d}.csv"
    pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)
    # reshape after blend — restore tails
    for amp in [1.10, 1.20]:
        y2 = rescale(y, amp); y2 = softpos(y2, 1.25, 0.85, "pos")
        name = f"v59_arblend_w{int(w_ar*100):02d}_amp{int(amp*100)}_sp.csv"
        pd.DataFrame({"session": y2.index, "target_position": y2.values}).to_csv(OUT / name, index=False)

# Strategy C: Use BOTH tails (compress low+high)
print("=== both tails ===")
for amp in [1.10, 1.20, 1.30]:
    for t in [1.25, 1.50]:
        for m in [0.75, 0.85]:
            y = rescale(b, amp); y = softpos(y, t, m, "both")
            name = f"v59_both_a{int(amp*100)}_t{int(t*100)}_m{int(m*100):03d}.csv"
            pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Strategy D: "Clip high" — flat cap at a level after reshape
print("=== clip high ===")
base = rescale(b, 1.10); base = softpos(base, 1.25, 0.85, "pos")
for cap in [2.5, 3.0, 3.5, 4.0]:
    y = base.clip(upper=cap)
    name = f"v59_cliphigh_cap{int(cap*10):02d}.csv"
    pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Strategy E: Skewness-reducer (compress all positive tails slightly)
# Raise power < 1 shrinks big positives more, but keeps sign
print("=== power map ===")
for p in [0.95, 0.90, 0.85]:
    y = np.sign(base - 1) * np.abs(base - 1) ** p + 1
    name = f"v59_pow_p{int(p*100):03d}.csv"
    pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

# Strategy F: Nonlinear amp — quadratic pullup
# y = base + k * (base - mu)^2 * sign(base - mu)
print("=== quadratic ===")
mu = base.mean()
for k in [0.05, 0.10, 0.15]:
    y = base + k * np.sign(base - mu) * (base - mu) ** 2
    name = f"v59_quad_k{int(k*100):03d}.csv"
    pd.DataFrame({"session": y.index, "target_position": y.values}).to_csv(OUT / name, index=False)

print("done")
EOF_MARKER_ignored = 0
