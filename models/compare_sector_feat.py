"""Paired-diff test: do the new sector-resolved event-impact features help?

Compares BASELINE (34 cols, no sector impacts) vs +SECTOR (40 cols) using
identical 40 random splits, so split-level noise cancels.
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

N_SPLITS = 40
HOLDOUT_FRAC = 0.2
ITERS = 52
BEST_KIND = "thresholded_inv_vol"

X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

print(f"Pre-building features across {N_SPLITS} splits ...")
prepped = []
for i, (hold_s, dev_s) in enumerate(splits):
    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]
    dev_event = build_event_features_oof(dev_h, dev_b, dev_s)
    split_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, split_impacts)

    dev_event_sec = build_event_features_sector_oof(dev_h, dev_b, dev_s)
    split_sec = fit_template_impacts_sector_multi(dev_h, dev_b)
    hold_event_sec = build_event_features_sector_multi(hold_h, hold_s, split_sec)

    Xd_base = X_base.loc[dev_s].join(dev_event)
    Xh_base = X_base.loc[hold_s].join(hold_event)
    Xd_full = Xd_base.join(dev_event_sec)
    Xh_full = Xh_base.join(hold_event_sec)
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]
    prepped.append((Xd_base, Xh_base, Xd_full, Xh_full, yd, yh))
    if (i + 1) % 10 == 0:
        print(f"  built {i+1}/{N_SPLITS}")

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

print("\nEvaluating baseline vs +sector ...")
res = {"baseline": [], "+sector": []}
for r, (Xd_b, Xh_b, Xd_f, Xh_f, yd, yh) in enumerate(prepped):
    res["baseline"].append(run(Xd_b, yd, Xh_b, yh, r))
    res["+sector"].append(run(Xd_f, yd, Xh_f, yh, r))

import numpy as np
base = np.array(res["baseline"])
sec = np.array(res["+sector"])
n = len(base)
for i, name in enumerate(["raw", "demeaned"]):
    b, s = base[:, i], sec[:, i]
    d = s - b
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se > 0 else 0.0
    print(f"  {name:>9}: base {b.mean():+.3f} ± {b.std(ddof=1)/np.sqrt(n):.3f}  |  "
          f"+sec {s.mean():+.3f} ± {s.std(ddof=1)/np.sqrt(n):.3f}  |  "
          f"paired Δ = {d.mean():+.3f} ± {se:.3f} (t={t:+.2f})")
