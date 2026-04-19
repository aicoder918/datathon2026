"""v33_best_tstpl_w20 → 2.74764 NEW BEST. Push ts_template_dir weight higher.
Also mix with other thinking-simple signals and aff2."""
import pandas as pd
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT_DIR = ROOT / "submissions"

best = pd.read_csv(OUT_DIR / "v32_best_aff2_w45.csv").set_index("session")["target_position"]
ts_tpl = pd.read_csv(OUT_DIR / "ts_template_dir.csv").set_index("session")["target_position"]
ts_proxy = pd.read_csv(OUT_DIR / "ts_proxy_fit.csv").set_index("session")["target_position"]
ts_lgbm = pd.read_csv(OUT_DIR / "ts_lgbm_featuregen.csv").set_index("session")["target_position"]
aff2 = pd.read_csv(OUT_DIR / "aff_tdvol_logstd_rdg85.csv").set_index("session")["target_position"]
v27 = pd.read_csv(OUT_DIR / "v27_meanens_top7.csv").set_index("session")["target_position"]

# Push ts_tpl weight higher — 2.74764 at w=20; peak likely 25-40
print("Finer ts_tpl sweep:")
for w in [0.12, 0.15, 0.17, 0.22, 0.25, 0.28, 0.32, 0.35, 0.40, 0.45, 0.50]:
    b = (1 - w) * best + w * ts_tpl
    name = f"v34_best_tstpl_w{int(w*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT_DIR / name, index=False)

# Triple: v27 + aff2 + ts_tpl (rebuild from scratch — maybe different split is better)
print("\nTriple v27 + aff2 + ts_tpl:")
for w_v, w_a, w_t in [
    (0.45, 0.35, 0.20), (0.40, 0.40, 0.20), (0.45, 0.30, 0.25),
    (0.40, 0.35, 0.25), (0.35, 0.40, 0.25), (0.50, 0.30, 0.20),
    (0.40, 0.30, 0.30), (0.35, 0.35, 0.30), (0.45, 0.25, 0.30),
    (0.30, 0.40, 0.30), (0.30, 0.35, 0.35), (0.35, 0.30, 0.35),
]:
    b = w_v * v27 + w_a * aff2 + w_t * ts_tpl
    name = f"v34_v27_{int(w_v*100)}_aff2_{int(w_a*100)}_tstpl_{int(w_t*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT_DIR / name, index=False)

# Quad: best + ts_tpl + ts_proxy + ts_lgbm
print("\nQuad best + 3 ts sources:")
for w_b, w_tp, w_tx, w_tl in [
    (0.75, 0.15, 0.05, 0.05),
    (0.70, 0.20, 0.05, 0.05),
    (0.70, 0.20, 0.10, 0.00),
    (0.65, 0.25, 0.05, 0.05),
    (0.60, 0.25, 0.10, 0.05),
    (0.65, 0.20, 0.10, 0.05),
    (0.70, 0.25, 0.00, 0.05),
    (0.65, 0.30, 0.00, 0.05),
]:
    b = w_b * best + w_tp * ts_tpl + w_tx * ts_proxy + w_tl * ts_lgbm
    name = f"v34_quad_b{int(w_b*100)}_ttpl{int(w_tp*100)}_tprx{int(w_tx*100)}_tlgb{int(w_tl*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT_DIR / name, index=False)

print("done")
