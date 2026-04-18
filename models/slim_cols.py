"""Paired-diff: which column groups can we drop from the 39-col production set?

Baseline = full 39 cols (bars + sentiment + 6 tid impact + 6 sector impact).
Variants drop groups; paired across N_SPLITS so noise cancels.
"""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor
from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_sector_multi, build_event_features_sector_multi,
    build_event_features_sector_oof,
    sharpe, shape_positions,
)

N_SPLITS = 120  # split 60/60 for pick-vs-confirm
HOLDOUT_FRAC = 0.2
ITERS = 52
BEST_KIND = "thresholded_inv_vol"

TID_COLS = [f"event_impact_k{k}" for k in (3, 5, 10)] + \
           [f"event_impact_recent_k{k}" for k in (3, 5, 10)]
SEC_COLS = [f"event_impact_sec_k{k}" for k in (3, 5, 10)] + \
           [f"event_impact_sec_recent_k{k}" for k in (3, 5, 10)]
RECENT_TID = [f"event_impact_recent_k{k}" for k in (3, 5, 10)]
RECENT_SEC = [f"event_impact_sec_recent_k{k}" for k in (3, 5, 10)]
SENT_COLS = ["hl_net_sent", "hl_net_sent_recent", "hl_mean_sent", "hl_n_pos", "hl_n_neg"]

ALL_IMPACT = TID_COLS + SEC_COLS

def keep_only(keep):
    return [c for c in ALL_IMPACT if c not in keep]

VARIANTS = {
    # group drops
    "-sector":            SEC_COLS,
    "-tid":               TID_COLS,
    "-all_impact":        ALL_IMPACT,
    "-recent_variants":   RECENT_TID + RECENT_SEC,
    "-sentiment":         SENT_COLS,
    "-sector_recent":     RECENT_SEC,
    "-tid_recent":        RECENT_TID,
    # single-keep (drop all other impact cols, keep just one)
    "keep_tid_k5":        keep_only(["event_impact_k5"]),
    "keep_tid_k3":        keep_only(["event_impact_k3"]),
    "keep_tid_k10":       keep_only(["event_impact_k10"]),
    "keep_tid_rec_k5":    keep_only(["event_impact_recent_k5"]),
    "keep_sec_k5":        keep_only(["event_impact_sec_k5"]),
    "keep_sec_rec_k5":    keep_only(["event_impact_sec_recent_k5"]),
    "keep_tid_k5+sec_k5": keep_only(["event_impact_k5", "event_impact_sec_k5"]),
}

X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]


def run(Xd, yd, Xh, yh, r):
    m = CatBoostRegressor(iterations=ITERS, learning_rate=0.03, depth=5,
                          loss_function="MAE", random_seed=SEED + r, verbose=False)
    m.fit(Xd, yd)
    p = np.asarray(m.predict(Xh), dtype=float)
    vh = np.asarray(Xh["vol"].values, dtype=float)
    yh_arr = np.asarray(yh.values, dtype=float)
    raw = sharpe(shape_positions(p, vh, BEST_KIND), yh_arr)
    dm = sharpe(shape_positions(p - p.mean(), vh, BEST_KIND), yh_arr)
    return raw, dm


print(f"Pre-building full 39-col features across {N_SPLITS} splits ...")
prepped = []
for i, (hold_s, dev_s) in enumerate(splits):
    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]

    dev_event = build_event_features_oof(dev_h, dev_b, dev_s)
    split_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, split_impacts)

    dev_sec = build_event_features_sector_oof(dev_h, dev_b, dev_s)
    sec_impacts = fit_template_impacts_sector_multi(dev_h, dev_b)
    hold_sec = build_event_features_sector_multi(hold_h, hold_s, sec_impacts)

    Xd = X_base.loc[dev_s].join(dev_event).join(dev_sec)
    Xh = X_base.loc[hold_s].join(hold_event).join(hold_sec)
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]
    prepped.append((Xd, Xh, yd, yh))
    if (i + 1) % 10 == 0:
        print(f"  built {i+1}/{N_SPLITS}")

print(f"\nFull feature cols ({prepped[0][0].shape[1]}):")
print(list(prepped[0][0].columns))

print("\nBaseline (full 39-col) once per split ...")
base = []
for r, (Xd, Xh, yd, yh) in enumerate(prepped):
    base.append(run(Xd, yd, Xh, yh, r))
base = np.array(base)
b_raw, b_dm = base[:, 0], base[:, 1]
se_b_raw = b_raw.std(ddof=1) / np.sqrt(N_SPLITS)
se_b_dm = b_dm.std(ddof=1) / np.sqrt(N_SPLITS)
print(f"baseline raw {b_raw.mean():+.3f} ± {se_b_raw:.3f}"
      f"  demeaned {b_dm.mean():+.3f} ± {se_b_dm:.3f}")

HALF = N_SPLITS // 2
results = {}
print()
for name, drop_cols in VARIANTS.items():
    scores = []
    for r, (Xd, Xh, yd, yh) in enumerate(prepped):
        Xd_v = Xd.drop(columns=[c for c in drop_cols if c in Xd.columns])
        Xh_v = Xh.drop(columns=[c for c in drop_cols if c in Xh.columns])
        scores.append(run(Xd_v, yd, Xh_v, yh, r))
    scores = np.array(scores)
    d_raw_all = scores[:, 0] - b_raw
    d_dm_all = scores[:, 1] - b_dm
    se_r = d_raw_all.std(ddof=1) / np.sqrt(N_SPLITS)
    se_d = d_dm_all.std(ddof=1) / np.sqrt(N_SPLITS)
    tr = d_raw_all.mean() / se_r if se_r > 0 else 0.0
    td = d_dm_all.mean() / se_d if se_d > 0 else 0.0
    # split-half
    dA, dB = d_raw_all[:HALF], d_raw_all[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean()/seA if seA>0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean()/seB if seB>0 else 0.0
    n_cols = prepped[0][0].shape[1] - len(drop_cols)
    results[name] = (d_raw_all.mean(), se_r, tr, d_dm_all.mean(), se_d, td,
                     dA.mean(), tA, dB.mean(), tB, n_cols)
    print(f"{name:<22} ({n_cols:2d} cols)  "
          f"Δraw={d_raw_all.mean():+.3f}±{se_r:.3f}(t={tr:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f})  B:{dB.mean():+.3f}(t={tB:+.2f})  "
          f"Δdm t={td:+.2f}")

print("\nReliability filter — variants with Δraw > 0 on BOTH halves:")
for name, (mr, sr, tr_, md, sd, td_, dA, tA, dB, tB, nc) in results.items():
    if dA > 0 and dB > 0:
        print(f"  {name:<22} ({nc:2d} cols)  Δraw={mr:+.3f}(t={tr_:+.2f})  "
              f"A:{dA:+.3f}(t={tA:+.2f}) B:{dB:+.3f}(t={tB:+.2f})")
