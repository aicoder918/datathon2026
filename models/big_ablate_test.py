"""Paired-diff: BIG ablations, not 3-5 col swaps. Test whether stripping whole
feature groups transfers differently than small perturbations.

Variants (all REMOVALS from the current full feature set):
  baseline       — everything (18 bar + 9 hl + 12 event_impact = 39 cols)
  no_hl          — drop ALL 9 headline features
  no_evt         — drop ALL 12 event_impact cols (base + sector)
  no_evt_sec     — drop ONLY the 6 sector event_impact cols
  no_hl_no_evt   — bars-only (18 cols)
  slim_bars      — drop near-constant price levels too (14 cols)
  core7          — vol, vol_recent, seen_ret, mom_5, mom_10,
                   max_drawdown, dist_to_high (7 cols)
  mini3          — vol, seen_ret, mom_5 (3 cols)

Hypothesis: if small perturbations fail to transfer because they fit 1k-train
idiosyncrasies, aggressive REMOVAL should either (a) hurt local & LB together
(features were signal) or (b) look slightly worse locally but equal/better on
LB (we stripped overfitting capacity). Any tier where local drop is small but
halves stay aligned is a candidate for an LB check.
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

N_SPLITS = 60
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
N_SEEDS = 3
CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)

HL_COLS = ["hl_n", "hl_n_recent", "hl_last_bar", "hl_mean_bar",
           "hl_net_sent", "hl_net_sent_recent", "hl_mean_sent",
           "hl_n_pos", "hl_n_neg"]
EVT_COLS = [f"event_impact_k{K}" for K in (3, 5, 10)] + \
           [f"event_impact_recent_k{K}" for K in (3, 5, 10)]
EVT_SEC_COLS = [f"event_impact_sec_k{K}" for K in (3, 5, 10)] + \
               [f"event_impact_sec_recent_k{K}" for K in (3, 5, 10)]
NEAR_CONST = ["close_first", "close_last", "max_high", "min_low"]

CORE7 = ["vol", "vol_recent", "seen_ret", "mom_5", "mom_10",
         "max_drawdown", "dist_to_high"]
MINI3 = ["vol", "seen_ret", "mom_5"]


def variant_cols(all_cols: list[str], variant: str) -> list[str]:
    """Return the subset of columns to keep for the named variant."""
    cols = set(all_cols)
    if variant == "baseline":
        return list(all_cols)
    if variant == "no_hl":
        cols -= set(HL_COLS)
    elif variant == "no_evt":
        cols -= set(EVT_COLS) | set(EVT_SEC_COLS)
    elif variant == "no_evt_sec":
        cols -= set(EVT_SEC_COLS)
    elif variant == "no_hl_no_evt":
        cols -= set(HL_COLS) | set(EVT_COLS) | set(EVT_SEC_COLS)
    elif variant == "slim_bars":
        cols -= set(HL_COLS) | set(EVT_COLS) | set(EVT_SEC_COLS) | set(NEAR_CONST)
    elif variant == "core7":
        cols = set(CORE7) & cols
    elif variant == "mini3":
        cols = set(MINI3) & cols
    else:
        raise ValueError(variant)
    # preserve original ordering
    return [c for c in all_cols if c in cols]


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def score_from_pred(pred, vol_h, y_h):
    pos = shape_positions(pred, vol_h, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    return sharpe(pos, y_h)


X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

VARIANTS = ["baseline", "no_hl", "no_evt", "no_evt_sec",
            "no_hl_no_evt", "slim_bars", "core7", "mini3"]
scores = {v: [] for v in VARIANTS}

print(f"Paired-diff: {N_SPLITS} splits × {N_SEEDS} seeds, {len(VARIANTS)} variants ...")
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

    all_cols = list(Xd_full.columns)
    for v in VARIANTS:
        keep = variant_cols(all_cols, v)
        Xd = Xd_full[keep]
        Xh = Xh_full[keep]
        pr = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
        scores[v].append(score_from_pred(pr, vh, yh))
    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}")

base = np.asarray(scores["baseline"])
print(f"\nbaseline raw   mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
print(f"\n{'variant':<14} {'ncols':>6}   {'mean':>7}   {'Δ vs base':>10}   {'t':>6}   halves (A/B)")
all_cols_full = list(splits[0][0])  # dummy, we don't actually use this var
# Use last loop's all_cols for ncols
for v in VARIANTS[1:]:
    s = np.asarray(scores[v])
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else (
           "!" if dA.mean() < 0 and dB.mean() < 0 else " ")
    ncols = len(variant_cols(all_cols, v))
    print(f"  {v:<12} {ncols:>4}   mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")

print("\nReading:")
print("  '*' both halves beat baseline → stripped features were hurting (strip-safe candidate)")
print("  '!' both halves lose to baseline → stripped features carried real signal")
print("  ' ' halves disagree → within split noise")
