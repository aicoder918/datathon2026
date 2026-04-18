"""Diagnostic: (1) correlation of every feature with target y, (2) feature-feature
correlation matrix, (3) paired-diff ablation variants dropping suspected-redundant
or near-constant features.

Ablation drops to test (each as a separate variant):
  -raw_levels    — drop close_first, close_last, max_high, min_low  (near-constants)
  -mom_short     — drop mom_1, mom_3 (keep mom_5, mom_10; seen_ret covers global)
  -hl_counts     — drop hl_n_pos, hl_n_neg (keep hl_n, hl_mean_sent carries sign)
  -combo         — all three drops combined
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_sector_multi, build_event_features_sector_multi,
    build_event_features_sector_oof,
    sharpe, shape_positions,
)

X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
print(f"Train sessions: {len(all_sessions)}  |  base cols: {list(X_base.columns)}")

# ---------- 1. correlation with target ----------
print("\n=== Feature → target Pearson correlation (sorted by |r|) ===")
corrs = {}
for c in X_base.columns:
    x = X_base[c].to_numpy(dtype=float)
    if np.std(x) < 1e-12:
        corrs[c] = 0.0; continue
    corrs[c] = float(np.corrcoef(x, y_full.to_numpy(dtype=float))[0, 1])
for c, r in sorted(corrs.items(), key=lambda kv: -abs(kv[1])):
    std = X_base[c].std()
    flag = "  <-- near-constant" if std < 1e-6 else ""
    print(f"  {c:<22}  r={r:+.4f}   std={std:.5f}{flag}")

# ---------- 2. cross-feature correlation (highlight >|0.9| pairs) ----------
print("\n=== Feature-pairs with |r| > 0.9 (redundancy hints) ===")
C = X_base.corr().fillna(0.0)
cols = list(C.columns)
seen_pairs = set()
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        if i >= j: continue
        r = C.iloc[i, j]
        if abs(r) > 0.9 and (b, a) not in seen_pairs:
            seen_pairs.add((a, b))
            print(f"  {a:<22} ~ {b:<22}  r={r:+.4f}")

# ---------- 3. paired-diff ablations ----------
N_SPLITS = 60
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
N_SEEDS = 3
CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def score_from_pred(pred, vol_h, y_h):
    pos = shape_positions(pred, vol_h, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    return sharpe(pos, y_h)


n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

DROP_VARIANTS = {
    "baseline":    [],
    "-raw_levels": ["close_first", "close_last", "max_high", "min_low"],
    "-mom_short":  ["mom_1", "mom_3"],
    "-hl_counts":  ["hl_n_pos", "hl_n_neg"],
    "-combo":      ["close_first", "close_last", "max_high", "min_low",
                    "mom_1", "mom_3", "hl_n_pos", "hl_n_neg"],
}
scores = {k: [] for k in DROP_VARIANTS}

print(f"\nPaired-diff ablations: {N_SPLITS} splits × {N_SEEDS} seeds ...")
for r, (hold_s, dev_s) in enumerate(splits):
    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]
    dev_event = build_event_features_oof(dev_h, dev_b, dev_s)
    split_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, split_impacts)
    dev_sec = build_event_features_sector_oof(dev_h, dev_b, dev_s)
    sec_impacts = fit_template_impacts_sector_multi(dev_h, dev_b)
    hold_sec = build_event_features_sector_multi(hold_h, hold_s, sec_impacts)
    Xd_full = X_base.loc[dev_s].join(dev_event).join(dev_sec)
    Xh_full = X_base.loc[hold_s].join(hold_event).join(hold_sec)
    yd = y_full.loc[dev_s].to_numpy(dtype=float)
    yh = y_full.loc[hold_s].to_numpy(dtype=float)
    vh = np.asarray(Xh_full["vol"].values, dtype=float)

    for name, drops in DROP_VARIANTS.items():
        Xd = Xd_full.drop(columns=drops) if drops else Xd_full
        Xh = Xh_full.drop(columns=drops) if drops else Xh_full
        pr = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
        scores[name].append(score_from_pred(pr, vh, yh))
    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}")

base = np.asarray(scores["baseline"])
print(f"\nbaseline raw  mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
print("\nDrop-variant results:")
for name in list(DROP_VARIANTS.keys())[1:]:
    s = np.asarray(scores[name])
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "
    print(f"  {name:<14} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")
