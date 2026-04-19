"""Blend current best with thinking-simple branch submissions.
Correlations are 0.07-0.83 with our best → huge potential diversity.
Submit individual thinking-simple first to know their LB, then blend carefully."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT_DIR = ROOT / "submissions"

best = pd.read_csv(OUT_DIR / "v32_best_aff2_w45.csv").set_index("session")["target_position"]
ts_lgbm = pd.read_csv(OUT_DIR / "ts_lgbm_featuregen.csv").set_index("session")["target_position"]
ts_proxy = pd.read_csv(OUT_DIR / "ts_proxy_fit.csv").set_index("session")["target_position"]
ts_template_dir = pd.read_csv(OUT_DIR / "ts_template_dir.csv").set_index("session")["target_position"]
# ts_aggressive excluded — corr 0.067 is TOO low, likely weak signal
# ts_template excluded — likely dominated by ts_template_dir

# Best + ts_lgbm (corr 0.273, very diverse)
print("Best + ts_lgbm:")
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    b = (1 - w) * best + w * ts_lgbm
    name = f"v33_best_tslgbm_w{int(w*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT_DIR / name, index=False)

# Best + ts_proxy (corr 0.825, diverse signal)
print("Best + ts_proxy:")
for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    b = (1 - w) * best + w * ts_proxy
    name = f"v33_best_tsproxy_w{int(w*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT_DIR / name, index=False)

# Best + ts_template_dir (corr 0.812)
print("Best + ts_template_dir:")
for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
    b = (1 - w) * best + w * ts_template_dir
    name = f"v33_best_tstpl_w{int(w*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT_DIR / name, index=False)

# Multi-source diverse: best + ts_proxy + ts_lgbm
print("Best + ts_proxy + ts_lgbm:")
for w_b, w_p, w_l in [(0.80, 0.10, 0.10), (0.75, 0.15, 0.10), (0.70, 0.20, 0.10),
                       (0.75, 0.10, 0.15), (0.70, 0.15, 0.15)]:
    b = w_b * best + w_p * ts_proxy + w_l * ts_lgbm
    name = f"v33_b{int(w_b*100)}_tsprox{int(w_p*100)}_tslgbm{int(w_l*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT_DIR / name, index=False)
    print(f"  {name}: mean={b.mean():.3f}")

print("done")
