"""Blend current best (aff7_amp140_softpos_pos_t125_m085, LB 2.86777) with
autoresearch signals — these have corr only 0.59 with best, the most diverse
signal source we've found yet. v17_finalize and v9 are different ML architectures
(Ridge mean + log-linear variance head) so offer real orthogonal info."""
import pandas as pd
from pathlib import Path

ROOT = Path("/home/misimon/datathon2026")
OUT = ROOT / "submissions"
AR = ROOT / "autoresearch" / "submissions"

best = pd.read_csv(ROOT / "best" / "best.csv").set_index("session")["target_position"]

signals = {
    "ar_v17":   pd.read_csv(AR / "v17_finalize.csv").set_index("session")["target_position"],
    "ar_v18a":  pd.read_csv(AR / "v18_a05_fl05.csv").set_index("session")["target_position"],
    "ar_v18n":  pd.read_csv(AR / "v18_finalize_nogamma.csv").set_index("session")["target_position"],
    "ar_v9":    pd.read_csv(AR / "v9_g003_lam07.csv").set_index("session")["target_position"],
    "ar_v13":   pd.read_csv(AR / "v13_blend_v7v9.csv").set_index("session")["target_position"],
    "ar_v15":   pd.read_csv(AR / "v15_richvar.csv").set_index("session")["target_position"],
    "ar_v14":   pd.read_csv(AR / "v14_prior30_tau25.csv").set_index("session")["target_position"],
    "ar_v12":   pd.read_csv(AR / "v12_ensemble_v7v9v11.csv").set_index("session")["target_position"],
}

# Conservative binary blends — low-corr signals need small weights
print("Binary blends (best + ar signals):")
weights = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
for tag, sig in signals.items():
    sig = sig.reindex(best.index)
    for w in weights:
        blend = (1 - w) * best + w * sig
        name = f"v44_best_{tag}_w{int(w*100):02d}.csv"
        pd.DataFrame({"session": blend.index, "target_position": blend.values}).to_csv(OUT / name, index=False)

# Mean of 4 most diverse AR signals
ar_mean4 = (signals["ar_v17"] + signals["ar_v9"] + signals["ar_v18n"] + signals["ar_v14"]) / 4
for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    blend = (1 - w) * best + w * ar_mean4.reindex(best.index)
    name = f"v44_best_armean4_w{int(w*100):02d}.csv"
    pd.DataFrame({"session": blend.index, "target_position": blend.values}).to_csv(OUT / name, index=False)

# Triple: best + aff2 (mid corr) + ar (low corr) — multi-mechanism
aff2 = pd.read_csv(OUT / "aff_tdvol_logstd_rdg85.csv").set_index("session")["target_position"].reindex(best.index)
for w_aff2, w_ar in [(0.10, 0.05), (0.10, 0.10), (0.15, 0.05), (0.15, 0.10), (0.20, 0.05)]:
    w_b = 1 - w_aff2 - w_ar
    blend = w_b * best + w_aff2 * aff2 + w_ar * signals["ar_v17"].reindex(best.index)
    name = f"v44_trip_b{int(w_b*100)}_aff2{int(w_aff2*100)}_arv17{int(w_ar*100)}.csv"
    pd.DataFrame({"session": blend.index, "target_position": blend.values}).to_csv(OUT / name, index=False)

print("done")
