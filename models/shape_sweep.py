"""Sweep position-shaping hyperparameters with cached predictions.

Hold the model fixed (39-col production config, CatBoost with ITERS=52).
Generate holdout predictions once per split, then sweep cheaply over:
  - threshold_q  (quantile cutoff for zeroing small preds)
  - vol_exp      (divide by vol**exp; 0 = no inv-vol, 1 = current)
  - shrink_alpha (blend toward constant long)
  - short_floor  (clamp shorts at this level)

Baseline = current production shaping: tq=0.25, vex=1.0, alpha=0.7, floor=0.0.
Report paired Δraw vs baseline across N_SPLITS.
"""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor
from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_sector_multi, build_event_features_sector_multi,
    build_event_features_sector_oof,
    sharpe,
)

N_SPLITS = 120
HOLDOUT_FRAC = 0.2
ITERS = 52

# baseline shaping
B_TQ = 0.25
B_VEX = 1.0
B_ALPHA = 0.7
B_FLOOR = 0.0

# sweep grids (each axis swept holding others at baseline)
TQ_GRID = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
VEX_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
ALPHA_GRID = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FLOOR_GRID = [-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.5]


def shape_and_finalize(pred, vol, tq, vex, alpha, floor):
    """shape_positions(thresholded_inv_vol) with vol_exp, then finalize."""
    cutoff = np.quantile(np.abs(pred), tq) if tq > 0 else 0.0
    out = pred / np.maximum(vol, 1e-6) ** vex if vex > 0 else pred.copy()
    out[np.abs(pred) < cutoff] = 0.0
    m = np.mean(np.abs(out))
    scaled = out / m if m > 0 else out
    blended = alpha * scaled + (1 - alpha) * 1.0
    return np.maximum(blended, floor)


X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

print(f"Pre-building 39-col features + cached preds across {N_SPLITS} splits ...")
cache = []  # (pred, vol, yh)
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

    m = CatBoostRegressor(iterations=ITERS, learning_rate=0.03, depth=5,
                          loss_function="MAE", random_seed=SEED + i, verbose=False)
    m.fit(Xd, yd)
    pred = np.asarray(m.predict(Xh), dtype=float)
    vol = np.asarray(Xh["vol"].values, dtype=float)
    yh_arr = np.asarray(yh.values, dtype=float)
    cache.append((pred, vol, yh_arr))
    if (i + 1) % 10 == 0:
        print(f"  built {i+1}/{N_SPLITS}")


def score(tq, vex, alpha, floor):
    return np.array([
        sharpe(shape_and_finalize(p, v, tq, vex, alpha, floor), y)
        for p, v, y in cache
    ])


HALF = N_SPLITS // 2
base = score(B_TQ, B_VEX, B_ALPHA, B_FLOOR)
print(f"\nbaseline (tq={B_TQ}, vex={B_VEX}, alpha={B_ALPHA}, floor={B_FLOOR}): "
      f"raw {base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")


def report(label, s):
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS)
    t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean()/seA if seA>0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean()/seB if seB>0 else 0.0
    both_pos = "*" if dA.mean() > 0 and dB.mean() > 0 else " "
    return (f"{label:<40} Δraw={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
            f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {both_pos}")


print("\n=== threshold_q sweep (vex=1, alpha=0.7, floor=0) ===")
for tq in TQ_GRID:
    print(report(f"tq={tq}", score(tq, B_VEX, B_ALPHA, B_FLOOR)))

print("\n=== vol_exp sweep (tq=0.25, alpha=0.7, floor=0) ===")
for vex in VEX_GRID:
    print(report(f"vex={vex}", score(B_TQ, vex, B_ALPHA, B_FLOOR)))

print("\n=== shrink_alpha sweep (tq=0.25, vex=1, floor=0) ===")
for alpha in ALPHA_GRID:
    print(report(f"alpha={alpha}", score(B_TQ, B_VEX, alpha, B_FLOOR)))

print("\n=== short_floor sweep (tq=0.25, vex=1, alpha=0.7) ===")
for fl in FLOOR_GRID:
    print(report(f"floor={fl}", score(B_TQ, B_VEX, B_ALPHA, fl)))

print("\n=== joint top-5 from coarse full grid ===")
rows = []
for tq in [0.0, 0.15, 0.25, 0.35]:
    for vex in [0.5, 1.0, 1.5]:
        for alpha in [0.5, 0.7, 0.9]:
            for fl in [0.0, 0.2, 0.3]:
                s = score(tq, vex, alpha, fl)
                d = s - base
                se = d.std(ddof=1) / np.sqrt(N_SPLITS)
                t = d.mean() / se if se > 0 else 0.0
                dA, dB = d[:HALF], d[HALF:]
                rows.append((t, d.mean(), se, tq, vex, alpha, fl,
                             dA.mean(), dB.mean()))
rows.sort(key=lambda r: -r[0])
print(f"{'t':>6} {'Δmean':>7} {'SE':>6}  tq    vex   alpha  floor   A       B    both+")
for t, dm, se, tq, vex, alpha, fl, dA, dB in rows[:15]:
    mark = "*" if dA > 0 and dB > 0 else " "
    print(f"{t:+6.2f} {dm:+7.3f} {se:6.3f}  {tq:4.2f}  {vex:4.2f}  {alpha:4.2f}  {fl:+5.2f}  "
          f"{dA:+6.3f}  {dB:+6.3f}  {mark}")
