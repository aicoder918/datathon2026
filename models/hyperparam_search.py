"""Random search over CatBoost hyperparams + position-shaping + finalize params.

Objective: mean RAW Sharpe across K held-out splits (the actual submission metric).

CatBoost fits are the expensive step; each fit yields a prediction vector per
split, then a dense grid over the cheap post-processing params (shape kind,
threshold, shrink_alpha, short_floor) picks the best knob setting for free.
"""
from __future__ import annotations
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi,
    build_event_features_oof, sharpe, shape_positions, finalize,
)

ROOT = Path(__file__).resolve().parent.parent

N_SPLITS = 5          # held-out evaluation splits per trial (lower-variance)
HOLDOUT_FRAC = 0.2
N_TRIALS = 30
RNG_SEED = SEED

# ---------- search spaces ----------
CAT_SPACE = {
    "depth": [4, 5, 6, 7],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    "iterations": [300, 600, 900, 1200],
}

# Cheap post-processing grid — explored after each fit without retraining.
# Include small alphas so the search can tell us how much the model tilt is
# actually worth (alpha=0 is the always-long degenerate baseline; we report it
# separately but it reflects "don't use the model").
KINDS = ["sign", "inv_vol", "thresholded", "thresholded_inv_vol"]
THRESHOLDS = [0.1, 0.25, 0.4]
SHRINK_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
SHORT_FLOORS = [0.0, -0.5, -1.0]


def sample_cat_config(rng: np.random.Generator) -> dict:
    return {k: rng.choice(v).item() for k, v in CAT_SPACE.items()}


def evaluate_predictions(pred_lookup: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, dict]:
    """Given a list of (preds, vol, y) per split, grid-search post-processing.
    Returns (best_mean_sharpe, best_config)."""
    best_s, best_cfg = -1e9, None
    for kind in KINDS:
        # thresholded kinds take threshold_q; others don't vary with it
        threshold_iter = THRESHOLDS if "threshold" in kind else [0.25]
        for tq in threshold_iter:
            for alpha in SHRINK_ALPHAS:
                for floor in SHORT_FLOORS:
                    sharpes = []
                    for pred, vol, y in pred_lookup:
                        pos = shape_positions(pred, vol, kind, threshold_q=tq)
                        pos = finalize(pos, shrink_alpha=alpha, short_floor=floor)
                        sharpes.append(sharpe(pos, y))
                    m = float(np.mean(sharpes))
                    if m > best_s:
                        best_s = m
                        best_cfg = {"kind": kind, "threshold_q": tq,
                                    "shrink_alpha": alpha, "short_floor": floor,
                                    "split_sharpes": [round(s, 3) for s in sharpes]}
    return best_s, best_cfg


# ---------- load base data once ----------
print("Loading base features ...")
X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
print(f"  {len(all_sessions)} sessions, {X_base.shape[1]} base cols")


# ---------- prebuild splits (features identical for every trial) ----------
rng = np.random.default_rng(RNG_SEED)
splits = []
for i in range(N_SPLITS):
    sh = rng.permutation(all_sessions)
    n_hold = int(len(sh) * HOLDOUT_FRAC)
    hold_s = np.sort(sh[:n_hold])
    dev_s = np.sort(sh[n_hold:])

    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]
    dev_event_oof = build_event_features_oof(dev_h, dev_b, dev_s)
    dev_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, dev_impacts)

    Xd = X_base.loc[dev_s].join(dev_event_oof)
    Xh = X_base.loc[hold_s].join(hold_event)
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]
    splits.append({
        "Xd": Xd, "yd": yd,
        "Xh": Xh, "yh_arr": np.asarray(yh.values, dtype=float),
        "vh": np.asarray(Xh["vol"].values, dtype=float),
    })
print(f"Built {N_SPLITS} splits. Feature count: {splits[0]['Xd'].shape[1]}")

# Reference point: constant-long ("always 1.0") Sharpe on these exact splits.
const_sharpes = [sharpe(np.ones_like(s["yh_arr"]), s["yh_arr"]) for s in splits]
print(f"Reference: constant-long mean Sharpe = {np.mean(const_sharpes):+.3f} "
      f"(per split: {[round(x, 2) for x in const_sharpes]})")


# ---------- trial loop ----------
log = []
best_score, best_trial = -1e9, None
t0 = time.time()
rng_trials = np.random.default_rng(RNG_SEED + 1)

# Seed trial 0 with the current-production params so we always have a baseline.
baseline_cat = {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 3.0, "iterations": 600}

for trial_i in range(N_TRIALS):
    cat_cfg = baseline_cat if trial_i == 0 else sample_cat_config(rng_trials)
    t_trial = time.time()
    pred_pack = []
    for s in splits:
        m = CatBoostRegressor(
            iterations=cat_cfg["iterations"],
            learning_rate=cat_cfg["learning_rate"],
            depth=cat_cfg["depth"],
            l2_leaf_reg=cat_cfg["l2_leaf_reg"],
            loss_function="RMSE", random_seed=SEED, verbose=False,
        )
        m.fit(s["Xd"], s["yd"])
        p = np.asarray(m.predict(s["Xh"]), dtype=float)
        pred_pack.append((p, s["vh"], s["yh_arr"]))

    score, post_cfg = evaluate_predictions(pred_pack)
    elapsed = time.time() - t_trial
    log.append({"trial": trial_i, "cat": cat_cfg, "post": post_cfg,
                "mean_raw_sharpe": round(score, 4), "sec": round(elapsed, 1)})
    tag = " ★" if score > best_score else ""
    print(f"[{trial_i:2d}] {elapsed:5.1f}s  "
          f"depth={cat_cfg['depth']} lr={cat_cfg['learning_rate']:.3f} "
          f"l2={cat_cfg['l2_leaf_reg']:>4.1f} it={cat_cfg['iterations']:>4d}  "
          f"→ kind={post_cfg['kind']:<22} α={post_cfg['shrink_alpha']:.1f} "
          f"floor={post_cfg['short_floor']:+.1f}  "
          f"sharpe={score:+.3f}{tag}")
    if score > best_score:
        best_score, best_trial = score, log[-1]


print(f"\nTotal search time: {(time.time() - t0)/60:.1f} min over {N_TRIALS} trials")
print("\n=== BEST TRIAL ===")
print(json.dumps(best_trial, indent=2))

# Persist log so runs are inspectable
with open(ROOT / "models" / "hyperparam_search_log.json", "w") as f:
    json.dump({"best": best_trial, "log": log}, f, indent=2)
print(f"\nSaved log to models/hyperparam_search_log.json")
