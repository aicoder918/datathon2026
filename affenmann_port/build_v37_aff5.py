"""Blend with affenmann's newest template044 and fine-grained template_dir variants.
Corr 0.84-0.85 — moderate diversity."""
import pandas as pd
from pathlib import Path

OUT = Path("/home/misimon/datathon2026/submissions")
best = pd.read_csv(OUT / "v33_best_tstpl_w20.csv").set_index("session")["target_position"]

sigs = {
    "t044r91":  pd.read_csv(OUT / "aff5_template044_ridge300_91_09.csv").set_index("session")["target_position"],
    "t044r93":  pd.read_csv(OUT / "aff5_template044_ridge300_93_07.csv").set_index("session")["target_position"],
    "t044r95":  pd.read_csv(OUT / "aff5_template044_ridge300_95_05.csv").set_index("session")["target_position"],
    "dir044":   pd.read_csv(OUT / "aff5_template_dir_a044_b159.csv").set_index("session")["target_position"],
    "dir046":   pd.read_csv(OUT / "aff5_template_dir_a046_b152.csv").set_index("session")["target_position"],
    "dir048":   pd.read_csv(OUT / "aff5_template_dir_a048_b146.csv").set_index("session")["target_position"],
}

print("Binary blends with aff5 signals:")
for tag, sig in sigs.items():
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        b = (1 - w) * best + w * sig
        name = f"v37_best_{tag}_w{int(w*100)}.csv"
        pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

# Mean of all 6 aff5 signals at various weights
print("\nMean-of-6 aff5 stack:")
mean_aff5 = sum(sigs.values()) / 6
for w in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    b = (1 - w) * best + w * mean_aff5
    name = f"v37_best_aff5mean_w{int(w*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

# Mix best + aff5mean + ctx (from v35)
ctx = pd.read_csv(OUT / "aff4_template_ctx_meansign20_3.csv").set_index("session")["target_position"]
print("\nBest + aff5mean + ctx:")
for w_a, w_c in [(0.15, 0.05), (0.20, 0.05), (0.15, 0.10), (0.20, 0.10),
                 (0.25, 0.05), (0.25, 0.10), (0.30, 0.05)]:
    w_b = 1 - w_a - w_c
    b = w_b * best + w_a * mean_aff5 + w_c * ctx
    name = f"v37_b{int(w_b*100)}_a5m{int(w_a*100)}_ctx{int(w_c*100)}.csv"
    pd.DataFrame({"session": b.index, "target_position": b.values}).to_csv(OUT / name, index=False)

print("done")
