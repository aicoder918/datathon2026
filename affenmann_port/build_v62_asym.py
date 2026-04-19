"""v62: Asymmetric reshape. Base distribution has heavier neg tail (p01=-0.50, p99=2.24).
Positive softpos at 2.25 already applied in BEST. Try:
  - Harder pos cap (softpos at lower threshold)
  - Neg-side softpos/clamp
  - Asymmetric scaling (long bias)
  - Sharper softpos (smaller m_scale)"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT/"submissions"

# Start from pre-softpos reshape (v54's blend before softpos) to allow fresh tail shaping
v54 = pd.read_csv(OUT/"v54_amp110_sp_t125_m085.csv").set_index("session")["target_position"]
ar = pd.read_csv(ROOT/"autoresearch/submissions/ar_current.csv").set_index("session")["target_position"].reindex(v54.index)

def rescale(x, s):
    mu = x.mean(); return mu + s * (x - mu)
def softpos(pos, t, m_scale, sides="pos"):
    out = pos.copy(); th = 1 + t
    if sides in ("pos","both"):
        m = pos > th; out[m] = m_scale*pos[m] + (1 - m_scale)
    if sides in ("neg","both"):
        m = pos < -th; out[m] = m_scale*pos[m] - (1 - m_scale)
    return out
def asym_softpos(pos, t_pos, m_pos, t_neg, m_neg):
    out = pos.copy()
    mp = pos > 1+t_pos; out[mp] = m_pos*pos[mp] + (1-m_pos)
    mn = pos < -(1+t_neg); out[mn] = m_neg*pos[mn] - (1-m_neg)
    return out

count = 0
w = 0.15
blend = (1-w)*v54 + w*ar

# --- A: Asymmetric softpos, vary neg side
for amp in [1.18, 1.20, 1.22]:
    for t_neg, m_neg in [(1.25, 0.85), (1.25, 0.75), (1.50, 0.85), (1.75, 0.85), (2.00, 0.85)]:
        y = rescale(blend, amp)
        y = asym_softpos(y, 1.25, 0.85, t_neg, m_neg)
        name = f"v62_asym_a{int(amp*100):03d}_tn{int(t_neg*100):03d}_mn{int(m_neg*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- B: Long-bias (scale positive side of (y-1) more than negative side)
for amp in [1.18, 1.20, 1.22]:
    for lb_factor in [1.05, 1.10, 1.15]:  # how much MORE to amp the long side
        y = blend.copy()
        mu = y.mean()
        dev = y - mu
        dev_asym = np.where(dev > 0, amp * lb_factor * dev, amp * dev)
        y = pd.Series(mu + dev_asym, index=y.index)
        y = softpos(y, 1.25, 0.85, "pos")
        name = f"v62_longbias_a{int(amp*100):03d}_lb{int(lb_factor*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- C: sharper softpos (smaller m) — more aggressive compression of tails
for amp in [1.18, 1.20, 1.22, 1.25]:
    for t, m_scale in [(1.25, 0.70), (1.25, 0.75), (1.25, 0.65), (1.00, 0.80), (1.50, 0.80)]:
        y = rescale(blend, amp); y = softpos(y, t, m_scale, "pos")
        name = f"v62_sharper_a{int(amp*100):03d}_t{int(t*100):03d}_m{int(m_scale*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- D: Hard cap / piecewise clamp (test if capping outliers helps)
for amp in [1.18, 1.20, 1.22]:
    for cap_high, cap_low in [(3.0, -1.5), (2.8, -1.3), (3.2, -1.7), (3.0, -1.0), (2.5, -1.0)]:
        y = rescale(blend, amp); y = softpos(y, 1.25, 0.85, "pos")
        y = y.clip(lower=cap_low, upper=cap_high)
        name = f"v62_cap_a{int(amp*100):03d}_h{int(cap_high*100):03d}_l{int(cap_low*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

# --- E: different rescale base (mean targets 1.0 vs observed 0.95)
for amp in [1.18, 1.20, 1.22]:
    for new_mean in [0.95, 1.00, 1.05]:
        dev = blend - blend.mean()
        y = pd.Series(new_mean + amp * dev, index=blend.index)
        y = softpos(y, 1.25, 0.85, "pos")
        name = f"v62_shift_a{int(amp*100):03d}_m{int(new_mean*100):03d}.csv"
        pd.DataFrame({"session":y.index,"target_position":y.values}).to_csv(OUT/name,index=False)
        count += 1

print(f"wrote {count} v62 candidates")
# Sanity
for label, fn in [("asym t150 mn85", "v62_asym_a120_tn150_mn085.csv"),
                  ("longbias 110",   "v62_longbias_a120_lb110.csv"),
                  ("sharper m70",    "v62_sharper_a120_t125_m070.csv"),
                  ("cap 30 -15",     "v62_cap_a120_h300_l-150.csv"),
                  ("shift m100",     "v62_shift_a120_m100.csv")]:
    f = OUT/fn
    if f.exists():
        x = pd.read_csv(f)["target_position"]
        print(f"{label}: mean={x.mean():.4f} std={x.std():.4f} min={x.min():.3f} max={x.max():.3f} #>3={(x>3).sum()}")
