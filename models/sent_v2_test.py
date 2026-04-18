"""Replace redundant sentiment features with a principled 5-feature set.

Drop:  hl_net_sent, hl_net_sent_recent, hl_mean_sent   (sum and mean of signed)
Add:   hl_sent_zscore   — sum(signed) / sqrt(max(n,1))    (conviction metric)
       hl_sent_max_pos  — max positive signed in session
       hl_sent_min_neg  — min negative signed in session
       hl_sent_drift    — mean(last 10 bars) - mean(first 40)
       hl_sent_std      — std of signed scores

Variants:
  +replace_sent3   — drop 3 redundant, add 5 new (net +2 cols)
  +add_sent5_only  — keep old 3, just add the 5 new (net +5 cols)
  +drop_only       — drop the 3 redundant, add nothing (baseline − 3)
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

REDUNDANT_COLS = ["hl_net_sent", "hl_net_sent_recent", "hl_mean_sent"]


def build_sent_v2(hdf: pd.DataFrame, all_sessions: np.ndarray) -> pd.DataFrame:
    h = hdf.merge(SENT_MAP, left_on="headline", right_index=True, how="left")
    h["signed"] = h["signed"].fillna(0.0)
    out = pd.DataFrame(index=all_sessions)
    out.index.name = "session"
    g = h.groupby("session")
    n = g.size()
    s_sum = g["signed"].sum()
    out["hl_sent_zscore"] = s_sum / np.sqrt(n.clip(lower=1))
    out["hl_sent_max_pos"] = g["signed"].max().clip(lower=0)   # 0 if no positive
    out["hl_sent_min_neg"] = g["signed"].min().clip(upper=0)   # 0 if no negative
    # drift: mean(bars >=40) - mean(bars <40)
    recent = h[h["bar_ix"] >= 40].groupby("session")["signed"].mean()
    early = h[h["bar_ix"] < 40].groupby("session")["signed"].mean()
    out["hl_sent_drift"] = (recent - early)
    out["hl_sent_std"] = g["signed"].std()  # NaN if n<2
    return out.fillna(0.0).sort_index()


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def score_from_pred(pred, vol_h, y_h):
    pos = shape_positions(pred, vol_h, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    return sharpe(pos, y_h)


X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()

sent_v2 = build_sent_v2(headlines_train, all_sessions)
print("sent_v2 features:")
print(sent_v2.describe().T)
print("\ncorr with y:")
for c in sent_v2.columns:
    r = np.corrcoef(sent_v2[c], y_full)[0, 1]
    print(f"  {c:<22}  r={r:+.4f}")

n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

VARIANTS = ["baseline", "+replace_sent3", "+add_sent5_only", "-drop_only"]
scores = {v: [] for v in VARIANTS}

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

    Xd_add = Xd_full.join(sent_v2.loc[dev_s])
    Xh_add = Xh_full.join(sent_v2.loc[hold_s])
    Xd_replace = Xd_full.drop(columns=REDUNDANT_COLS).join(sent_v2.loc[dev_s])
    Xh_replace = Xh_full.drop(columns=REDUNDANT_COLS).join(sent_v2.loc[hold_s])
    Xd_drop = Xd_full.drop(columns=REDUNDANT_COLS)
    Xh_drop = Xh_full.drop(columns=REDUNDANT_COLS)

    for v, (Xd, Xh) in [
        ("baseline", (Xd_full, Xh_full)),
        ("+replace_sent3", (Xd_replace, Xh_replace)),
        ("+add_sent5_only", (Xd_add, Xh_add)),
        ("-drop_only", (Xd_drop, Xh_drop)),
    ]:
        pr = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
        scores[v].append(score_from_pred(pr, vh, yh))
    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}")

base = np.asarray(scores["baseline"])
print(f"\nbaseline raw     mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
for v in VARIANTS[1:]:
    s = np.asarray(scores[v])
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "
    print(f"  {v:<18} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")
