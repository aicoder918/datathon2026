"""Deep exploration of low-corr signals from aff4.
v33_best_tstpl_w20 jumped +0.026 via low-corr ts_template_dir.
Now test if aff4's ctx (corr 0.14) and zsum (corr 0.34) deliver similar jumps."""
import pandas as pd
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT / "submissions"

best = pd.read_csv(OUT / "v33_best_tstpl_w20.csv").set_index("session")["target_position"]
ts_tpl = pd.read_csv(OUT / "ts_template_dir.csv").set_index("session")["target_position"]

sigs = {
    "ctx":      pd.read_csv(OUT / "aff4_template_ctx_meansign20_3.csv").set_index("session")["target_position"],
    "ctxw":     pd.read_csv(OUT / "aff4_template_ctx_meansign20_3_winner_90_10.csv").set_index("session")["target_position"],
    "zsumw":    pd.read_csv(OUT / "aff4_template_zsum_dir49_ols_winner_95_05.csv").set_index("session")["target_position"],
    "tidsec":   pd.read_csv(OUT / "aff4_template_tidsec_ctx_meansign20_3_mean.csv").set_index("session")["target_position"],
    "zsummean": pd.read_csv(OUT / "aff4_template_zsum_meansign20_cap30_winner_90_10.csv").set_index("session")["target_position"],
    "dir060":   pd.read_csv(OUT / "aff4_template_dir_a060_b117.csv").set_index("session")["target_position"],
}

# Stack ALL diverse signals (small weights) — maximize diversity
print("Diverse stack on top of current best:")
# equal-weight small stack
for w_each in [0.03, 0.05, 0.07]:
    s = sum(sigs[k] for k in ["ctx","ctxw","zsumw","tidsec","zsummean"]) / 5
    b = (1 - 5*w_each) * best + 5*w_each * s
    name = f"v36_stack5_we{int(w_each*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

# Best + ctx + ts_tpl (double diverse) — MIGHT stack the tstpl effect
print("\nBest + ctx + extra ts_tpl:")
for w_ctx, w_extra_tpl in [(0.10, 0.05), (0.15, 0.05), (0.10, 0.10), (0.15, 0.10),
                            (0.20, 0.05), (0.20, 0.10), (0.15, 0.15)]:
    w_best = 1 - w_ctx - w_extra_tpl
    b = w_best * best + w_ctx * sigs["ctx"] + w_extra_tpl * ts_tpl
    name = f"v36_b{int(w_best*100)}_ctx{int(w_ctx*100)}_extpl{int(w_extra_tpl*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

# Rebuild v27_meanens_top7 + aff2 + ts_tpl + ctx (deeper ensemble)
v27 = pd.read_csv(OUT / "v27_meanens_top7.csv").set_index("session")["target_position"]
aff2 = pd.read_csv(OUT / "aff_tdvol_logstd_rdg85.csv").set_index("session")["target_position"]

print("\n4-way: v27 + aff2 + ts_tpl + ctx:")
for w_v, w_a, w_t, w_c in [
    (0.40, 0.30, 0.20, 0.10), (0.40, 0.30, 0.15, 0.15),
    (0.35, 0.30, 0.20, 0.15), (0.40, 0.25, 0.20, 0.15),
    (0.35, 0.35, 0.20, 0.10), (0.35, 0.30, 0.25, 0.10),
    (0.30, 0.30, 0.25, 0.15), (0.30, 0.35, 0.20, 0.15),
]:
    b = w_v*v27 + w_a*aff2 + w_t*ts_tpl + w_c*sigs["ctx"]
    name = f"v36_4way_{int(w_v*100)}_{int(w_a*100)}_{int(w_t*100)}_{int(w_c*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

# Fine tstpl at higher weights + ctx small boost
print("\nFine tstpl w + ctx small:")
for w_tpl in [0.22, 0.25, 0.28, 0.30]:
    for w_c in [0.03, 0.05, 0.08, 0.10]:
        w_b = 1 - w_tpl - w_c
        b = w_b * v27 + w_tpl * ts_tpl + w_c * sigs["ctx"]
        name = f"v36_v27_{int(w_b*100)}_ttpl{int(w_tpl*100)}_ctx{int(w_c*100)}.csv"
        pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

print("done")
