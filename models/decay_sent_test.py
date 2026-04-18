"""Paired-diff: add time-decayed sentiment features to the base 28-col set.

Candidates (all weighted by w = exp(-(49 - bar_ix) / tau)):
  +decay10   — one tau=10 feature: hl_decay_sent (sum w*signed)
  +decay5    — tau=5
  +decay20   — tau=20
  +decay_all — all three + mean_recent10
  +meanR10   — only hl_mean_sent_recent (mean signed over bars 40-49)

Same 60-split scaffolding as bar_mlm_emb, seed-3 ensemble per split.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import (
    SEED, SENT_MAP, load_train_base,
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


def build_decay_features(hdf: pd.DataFrame, all_sessions: np.ndarray,
                         taus=(5, 10, 20)) -> pd.DataFrame:
    """Per-session time-decayed sentiment sums + mean_recent10."""
    h = hdf.merge(SENT_MAP, left_on="headline", right_index=True, how="left")
    h["signed"] = h["signed"].fillna(0.0)
    out = pd.DataFrame(index=all_sessions)
    out.index.name = "session"
    for tau in taus:
        w = np.exp(-(49 - h["bar_ix"].to_numpy()) / tau)
        tmp = h.assign(wsig=w * h["signed"].to_numpy())
        out[f"hl_decay_sent_t{tau}"] = tmp.groupby("session")["wsig"].sum()
    # mean over last 10 bars
    recent = h[h["bar_ix"] >= 40]
    out["hl_mean_sent_recent10"] = recent.groupby("session")["signed"].mean()
    out = out.fillna(0.0)
    return out


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

# precompute full decay features (train only — using full train headlines; no leakage since
# these are per-session aggregates that don't depend on other sessions' targets)
decay_all = build_decay_features(headlines_train, all_sessions)
print("Decay features:", decay_all.columns.tolist())
print(decay_all.describe().T)

VARIANTS = {
    "baseline":  [],
    "+decay5":   ["hl_decay_sent_t5"],
    "+decay10":  ["hl_decay_sent_t10"],
    "+decay20":  ["hl_decay_sent_t20"],
    "+meanR10":  ["hl_mean_sent_recent10"],
    "+decay_all":["hl_decay_sent_t5", "hl_decay_sent_t10", "hl_decay_sent_t20",
                  "hl_mean_sent_recent10"],
}

scores = {k: [] for k in VARIANTS}
print(f"\nPaired-diff: {N_SPLITS} splits × {N_SEEDS} seeds, {len(VARIANTS)} variants ...")
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

    for name, extra_cols in VARIANTS.items():
        if extra_cols:
            Xd = Xd_full.join(decay_all.loc[dev_s, extra_cols])
            Xh = Xh_full.join(decay_all.loc[hold_s, extra_cols])
        else:
            Xd, Xh = Xd_full, Xh_full
        pr = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
        scores[name].append(score_from_pred(pr, vh, yh))

    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}")

base = np.asarray(scores["baseline"])
se_b = base.std(ddof=1) / np.sqrt(N_SPLITS)
print(f"\nbaseline raw    mean={base.mean():+.3f} ± {se_b:.3f}")

print("\nResults:")
for name in list(VARIANTS.keys())[1:]:
    s = np.asarray(scores[name])
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "
    print(f"  {name:<14} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")
