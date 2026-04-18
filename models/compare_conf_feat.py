"""Paired-diff: does the single confidence-weighted event_impact_conf feature help?

Compares:
  BASELINE:  34 cols (bars + sentiment + 6 event_impact)
  +CONF:     35 cols (baseline + 1 event_impact_conf)
  REPLACE:   29 cols (baseline WITHOUT 6 event_impact + 1 event_impact_conf)

All three share the same 40 random splits so paired noise cancels.
"""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor
from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_confidence, build_event_features_confidence,
    build_event_features_confidence_oof,
    sharpe, shape_positions,
)

N_SPLITS = 40
HOLDOUT_FRAC = 0.2
ITERS = 52
BEST_KIND = "thresholded_inv_vol"
IMPACT_COLS = [f"event_impact_k{k}" for k in (3, 5, 10)] + \
              [f"event_impact_recent_k{k}" for k in (3, 5, 10)]

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

    dev_conf = build_event_features_confidence_oof(dev_h, dev_b, dev_s)
    tid_imp, pair_imp = fit_template_impacts_confidence(dev_h, dev_b)
    hold_conf = build_event_features_confidence(hold_h, hold_s, tid_imp, pair_imp)

    Xd_base = X_base.loc[dev_s].join(dev_event)
    Xh_base = X_base.loc[hold_s].join(hold_event)
    Xd_plus = Xd_base.join(dev_conf)
    Xh_plus = Xh_base.join(hold_conf)
    Xd_repl = Xd_plus.drop(columns=IMPACT_COLS)
    Xh_repl = Xh_plus.drop(columns=IMPACT_COLS)
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]
    prepped.append((Xd_base, Xh_base, Xd_plus, Xh_plus, Xd_repl, Xh_repl, yd, yh))
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

print("\nEvaluating baseline / +conf / replace ...")
res = {"baseline": [], "+conf": [], "replace": []}
for r, (Xd_b, Xh_b, Xd_p, Xh_p, Xd_r, Xh_r, yd, yh) in enumerate(prepped):
    res["baseline"].append(run(Xd_b, yd, Xh_b, yh, r))
    res["+conf"].append(run(Xd_p, yd, Xh_p, yh, r))
    res["replace"].append(run(Xd_r, yd, Xh_r, yh, r))

base = np.array(res["baseline"])
for label, arr in [("+conf", np.array(res["+conf"])), ("replace", np.array(res["replace"]))]:
    print(f"\n{label} vs baseline:")
    for i, name in enumerate(["raw", "demeaned"]):
        b, s = base[:, i], arr[:, i]
        d = s - b
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else 0.0
        print(f"  {name:>9}: base {b.mean():+.3f} ± {b.std(ddof=1)/np.sqrt(len(b)):.3f}  |  "
              f"{label:<7} {s.mean():+.3f} ± {s.std(ddof=1)/np.sqrt(len(s)):.3f}  |  "
              f"paired Δ = {d.mean():+.3f} ± {se:.3f} (t={t:+.2f})")
