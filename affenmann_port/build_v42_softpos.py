"""Port softpos formula from reverse-engineering: above 1+t, new = m*pos + (1-m).
Best so far: t=1.25, m=0.85 → LB 2.86777. Sweep t and m to find peak."""
import pandas as pd, numpy as np
from pathlib import Path

OUT = Path("/home/misimon/datathon2026/submissions")

base_amp140 = pd.read_csv(OUT / "aff7_template044_ridge1000_58_42_amp140.csv").set_index("session")["target_position"]


def softpos(pos: pd.Series, t: float, m: float, sides: str = "pos") -> pd.Series:
    """Soft compression of tails.
    Above threshold (1+t): new = m*pos + (1-m). Mirror for negative if sides=='both'.
    """
    out = pos.copy()
    th = 1 + t
    if sides in ("pos", "both"):
        mask = pos > th
        out[mask] = m * pos[mask] + (1 - m)
    if sides in ("neg", "both"):
        mask = pos < -th
        out[mask] = m * pos[mask] - (1 - m)
    return out


print("=== softpos sweeps on amp140 ===")

# Sweep t (threshold) at fixed m=0.85, positive only
for t in [0.50, 0.75, 1.00, 1.10, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00]:
    s = softpos(base_amp140, t, 0.85, "pos")
    name = f"v42_amp140_sppos_t{int(t*100)}_m085.csv"
    pd.DataFrame({"session": s.index, "target_position": s.values}).to_csv(OUT / name, index=False)

# Sweep m (compression) at t=1.25, positive only
for m in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
    s = softpos(base_amp140, 1.25, m, "pos")
    name = f"v42_amp140_sppos_t125_m{int(m*100):03d}.csv"
    pd.DataFrame({"session": s.index, "target_position": s.values}).to_csv(OUT / name, index=False)

# Sweep BOTH sides (pos + neg)
for t, m in [(1.25, 0.85), (1.25, 0.80), (1.00, 0.85), (1.50, 0.85), (1.25, 0.75)]:
    s = softpos(base_amp140, t, m, "both")
    name = f"v42_amp140_spboth_t{int(t*100)}_m{int(m*100):03d}.csv"
    pd.DataFrame({"session": s.index, "target_position": s.values}).to_csv(OUT / name, index=False)

# Same on amp145 base
base_amp145 = pd.read_csv(OUT / "aff7_template044_ridge1000_58_42_amp145_softtail.csv").set_index("session")["target_position"]
# Note: aff7_amp145 doesn't have a non-softtail variant in our submissions, build from amp140 * 1.45/1.40
# Actually simpler: read straight from a softtail-undone version isn't possible; use the existing amp145_softtail
print("\n=== softpos on amp145 (already softtailed) ===")
for t, m in [(1.25, 0.85), (1.30, 0.85), (1.25, 0.80), (1.50, 0.85)]:
    s = softpos(base_amp145, t, m, "pos")
    name = f"v42_amp145_sppos_t{int(t*100)}_m{int(m*100):03d}.csv"
    pd.DataFrame({"session": s.index, "target_position": s.values}).to_csv(OUT / name, index=False)

# Stack: softpos then mild blend with current best
best = pd.read_csv("/home/misimon/datathon2026/best/best.csv").set_index("session")["target_position"]
print("\n=== best + small softpos perturbation ===")
for t, m, w in [(1.10, 0.85, 0.5), (1.10, 0.80, 0.5), (1.20, 0.80, 0.7)]:
    sp = softpos(base_amp140, t, m, "pos")
    blend = w * best + (1 - w) * sp
    name = f"v42_blend_best{int(w*100)}_sppos_t{int(t*100)}_m{int(m*100):03d}.csv"
    pd.DataFrame({"session": blend.index, "target_position": blend.values}).to_csv(OUT / name, index=False)

print("\ndone")
