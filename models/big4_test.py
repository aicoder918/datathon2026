"""Paired-diff: seed-ensemble + target transforms.

Variants vs baseline (1 seed, target=y, shape=thresholded_inv_vol):
  +seedN        — average predictions of N CatBoosts with different seeds
  +yvol         — train on y/vol_dev; at predict time multiply pred by vol_hold
                  to recover y-scale, then shape as usual
  +rank         — train on centered-rank(y) in dev set; use pred directly
  +seed5_yvol   — combine
  +seed5_rank   — combine

Same 120 splits, same per-split feature prep (reused across variants).
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

N_SPLITS = 120
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def score_from_pred(pred_y_scale, vol_h, y_h):
    pos = shape_positions(pred_y_scale, vol_h, "thresholded_inv_vol",
                          threshold_q=THRESHOLD_Q)
    return sharpe(pos, y_h)


# --- prep splits ---
X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

print(f"Pre-building per-split features across {N_SPLITS} splits ...")
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
    yd = y_full.loc[dev_s].to_numpy(dtype=float)
    yh = y_full.loc[hold_s].to_numpy(dtype=float)
    vol_h = np.asarray(Xh["vol"].values, dtype=float)
    vol_d = np.asarray(Xd["vol"].values, dtype=float)
    prepped.append((Xd, Xh, yd, yh, vol_d, vol_h))
    if (i + 1) % 20 == 0:
        print(f"  built {i+1}/{N_SPLITS}")


N_SEEDS = 5  # ensemble size we care about


# --- per split, precompute preds for N_SEEDS y-target fits (reused) ---
print(f"\nFitting baseline + ensemble CatBoost ({N_SEEDS} seeds per split) ...")
per_split_preds_y = []   # shape: [N_SPLITS][N_SEEDS] -> pred_h
for r, (Xd, Xh, yd, yh, vd, vh) in enumerate(prepped):
    preds = [cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)]
    per_split_preds_y.append(preds)
    if (r + 1) % 20 == 0:
        print(f"  fit {r+1}/{N_SPLITS}")


# --- target transform fits (only for k=1 and k=5 seeds to keep runtime ok) ---
print("\nFitting target=y/vol and target=rank ...")
per_split_preds_yvol = []   # pred is y/vol; we rescale by vol_h at use time
per_split_preds_rank = []   # pred is rank-target; use directly
for r, (Xd, Xh, yd, yh, vd, vh) in enumerate(prepped):
    y_over_v = yd / np.maximum(vd, 1e-6)
    yv = [cb_pred(Xd, y_over_v, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)]
    per_split_preds_yvol.append(yv)
    # centered rank in [-0.5, 0.5]
    ranks = (yd.argsort().argsort().astype(float)) / max(len(yd) - 1, 1) - 0.5
    yr = [cb_pred(Xd, ranks, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)]
    per_split_preds_rank.append(yr)
    if (r + 1) % 20 == 0:
        print(f"  fit {r+1}/{N_SPLITS}")


def variant_scores(kind, n_seeds):
    """kind in {'y','yvol','rank'}; n_seeds averaged."""
    scores = []
    for r, (Xd, Xh, yd, yh, vd, vh) in enumerate(prepped):
        if kind == "y":
            pr = np.mean(per_split_preds_y[r][:n_seeds], axis=0)
            scores.append(score_from_pred(pr, vh, yh))
        elif kind == "yvol":
            pr_yv = np.mean(per_split_preds_yvol[r][:n_seeds], axis=0)
            # rescale to y-scale so shape function sees comparable magnitudes
            pr = pr_yv * vh
            scores.append(score_from_pred(pr, vh, yh))
        elif kind == "rank":
            pr = np.mean(per_split_preds_rank[r][:n_seeds], axis=0)
            scores.append(score_from_pred(pr, vh, yh))
        else:
            raise ValueError(kind)
    return np.asarray(scores)


base = variant_scores("y", 1)
se_b = base.std(ddof=1) / np.sqrt(N_SPLITS)
print(f"\nbaseline raw {base.mean():+.3f} ± {se_b:.3f}")


def report(label, s):
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean()/se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean()/seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean()/seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "
    return (f"{label:<18} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
            f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")


variants = [
    ("+seed3", variant_scores("y", 3)),
    ("+seed5", variant_scores("y", 5)),
    ("+yvol",  variant_scores("yvol", 1)),
    ("+yvol_seed5", variant_scores("yvol", 5)),
    ("+rank",  variant_scores("rank", 1)),
    ("+rank_seed5", variant_scores("rank", 5)),
]
for label, s in variants:
    print(report(label, s))

print("\nReliability filter — both halves Δ>0:")
for label, s in variants:
    d = s - base
    dA, dB = d[:HALF], d[HALF:]
    if dA.mean() > 0 and dB.mean() > 0:
        se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean()/se if se > 0 else 0.0
        print(f"  {label:<18}  Δ={d.mean():+.3f}(t={t:+.2f})  "
              f"A:{dA.mean():+.3f}  B:{dB.mean():+.3f}")
