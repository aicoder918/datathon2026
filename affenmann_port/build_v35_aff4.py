"""Blend current best (v33_best_tstpl_w20 → 2.74764) with affenmann's newest submissions.
Massive diversity: corr 0.08-0.88. Biggest potential from template_ctx (corr 0.14) and
template_tidsec (corr 0.09)."""
import pandas as pd
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT / "submissions"

best = pd.read_csv(OUT / "v33_best_tstpl_w20.csv").set_index("session")["target_position"]

sigs = {
    "ctx":        pd.read_csv(OUT / "aff4_template_ctx_meansign20_3.csv").set_index("session")["target_position"],
    "ctxw":       pd.read_csv(OUT / "aff4_template_ctx_meansign20_3_winner_90_10.csv").set_index("session")["target_position"],
    "dir060":     pd.read_csv(OUT / "aff4_template_dir_a060_b117.csv").set_index("session")["target_position"],
    "zsumw":      pd.read_csv(OUT / "aff4_template_zsum_dir49_ols_winner_95_05.csv").set_index("session")["target_position"],
    "tidsec":     pd.read_csv(OUT / "aff4_template_tidsec_ctx_meansign20_3_mean.csv").set_index("session")["target_position"],
    "zsummean":   pd.read_csv(OUT / "aff4_template_zsum_meansign20_cap30_winner_90_10.csv").set_index("session")["target_position"],
}

# Conservative binary blends — low-corr signals at 5-20%, high-corr at 10-30%
print("Binary blends:")
grids = {
    "ctx":      [0.05, 0.10, 0.15, 0.20, 0.25],
    "ctxw":     [0.05, 0.10, 0.15, 0.20, 0.25],
    "dir060":   [0.10, 0.20, 0.30, 0.40],
    "zsumw":    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "tidsec":   [0.03, 0.05, 0.08, 0.12, 0.15],
    "zsummean": [0.05, 0.10, 0.15, 0.20, 0.25],
}
for tag, ws in grids.items():
    for w in ws:
        b = (1 - w) * best + w * sigs[tag]
        name = f"v35_best_{tag}_w{int(w*100)}.csv"
        pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

# Triple: best + ctx + zsumw (both diverse, different mechanisms)
print("\nTriple best + ctx + zsumw:")
for w_b, w_c, w_z in [(0.80, 0.10, 0.10), (0.75, 0.15, 0.10), (0.75, 0.10, 0.15),
                      (0.70, 0.15, 0.15), (0.70, 0.20, 0.10), (0.70, 0.10, 0.20),
                      (0.65, 0.20, 0.15), (0.65, 0.15, 0.20)]:
    b = w_b * best + w_c * sigs["ctx"] + w_z * sigs["zsumw"]
    name = f"v35_b{int(w_b*100)}_ctx{int(w_c*100)}_zsumw{int(w_z*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

# Quad: best + ctx + zsumw + tidsec (deepest diversity)
print("\nQuad best + 3 diverse:")
for w_b, w_c, w_z, w_t in [(0.80, 0.08, 0.08, 0.04), (0.75, 0.10, 0.10, 0.05),
                            (0.70, 0.12, 0.12, 0.06), (0.70, 0.15, 0.10, 0.05),
                            (0.72, 0.10, 0.12, 0.06), (0.70, 0.10, 0.15, 0.05)]:
    b = w_b * best + w_c * sigs["ctx"] + w_z * sigs["zsumw"] + w_t * sigs["tidsec"]
    name = f"v35_q{int(w_b*100)}_ctx{int(w_c*100)}_zsw{int(w_z*100)}_tid{int(w_t*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

print("done")
