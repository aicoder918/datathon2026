"""Verify top conf_feat configs with N_SPLITS=100 paired against baseline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_confidence, _attach_tid_sec,
    sharpe, shape_positions,
)

N_SPLITS = 100
HOLDOUT_FRAC = 0.2
ITERS = 52
BEST_KIND = "thresholded_inv_vol"
MAX_BAR = 50

CONFIGS = [
    dict(K=5, conf_k=30.0, bar_start=40, power=0.0),
    dict(K=3, conf_k=30.0, bar_start=40, power=1.0),
    dict(K=3, conf_k=10.0, bar_start=35, power=0.0),
    dict(K=5, conf_k=30.0, bar_start=35, power=0.0),
]

X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]


def contribs(h_tag, tid_stats, pair_stats):
    out = np.zeros(len(h_tag), dtype=float)
    tids = h_tag["_tid"].to_numpy(int)
    secs = h_tag["_sec"].to_numpy()
    for i, (t, s) in enumerate(zip(tids, secs)):
        stat = pair_stats.get((int(t), str(s))) or tid_stats.get(int(t))
        if stat is None:
            continue
        infl, conf = stat
        out[i] = infl * conf
    return out


def aggregate(h_tag, ic, sessions, bar_start, power):
    span = float(MAX_BAR - bar_start)
    bars = h_tag["bar_ix"].to_numpy(float)
    mask = (bars >= bar_start) & (bars < MAX_BAR)
    r = np.clip((bars - (bar_start - 1)) / span, 0.0, 1.0)
    vals = ic * (r ** power) * mask
    df = pd.DataFrame({"session": h_tag["session"].values, "v": vals})
    agg = df.groupby("session")["v"].sum()
    out = pd.Series(0.0, index=sessions)
    inter = agg.index.intersection(sessions)
    out.loc[inter] = agg.reindex(inter).values
    return out.values


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


print(f"Pre-building features across {N_SPLITS} splits ...")
prepped = []
for i, (hold_s, dev_s) in enumerate(splits):
    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]
    dev_event = build_event_features_oof(dev_h, dev_b, dev_s)
    split_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, split_impacts)
    Xd_base = X_base.loc[dev_s].join(dev_event)
    Xh_base = X_base.loc[hold_s].join(hold_event)
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]

    dev_h_tag = _attach_tid_sec(dev_h)
    dev_h_tag = dev_h_tag[dev_h_tag["_tid"] >= 0].reset_index(drop=True)
    hold_h_tag = _attach_tid_sec(hold_h)
    hold_h_tag = hold_h_tag[hold_h_tag["_tid"] >= 0].reset_index(drop=True)
    sess_arr = dev_h_tag["session"].to_numpy()
    session_to_idx = {s: np.where(sess_arr == s)[0] for s in dev_s}

    per_c = {}
    cache_kc = {}
    for cfg in CONFIGS:
        K, ck = cfg["K"], cfg["conf_k"]
        if (K, ck) not in cache_kc:
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            dev_ic = np.zeros(len(dev_h_tag), dtype=float)
            for tr_idx, va_idx in kf.split(dev_s):
                tr_s = dev_s[tr_idx]; va_s = dev_s[va_idx]
                h_tr = dev_h[dev_h["session"].isin(tr_s)]
                b_tr = dev_b[dev_b["session"].isin(tr_s)]
                t_stats, p_stats = fit_template_impacts_confidence(h_tr, b_tr, K=K, conf_k=ck)
                va_rows = np.concatenate([session_to_idx[s] for s in va_s if s in session_to_idx])
                if len(va_rows):
                    dev_ic[va_rows] = contribs(dev_h_tag.iloc[va_rows], t_stats, p_stats)
            t_stats, p_stats = fit_template_impacts_confidence(dev_h, dev_b, K=K, conf_k=ck)
            hold_ic = contribs(hold_h_tag, t_stats, p_stats)
            cache_kc[(K, ck)] = (dev_ic, hold_ic)

    prepped.append((Xd_base, Xh_base, yd, yh, dev_s, hold_s,
                    dev_h_tag, hold_h_tag, cache_kc))
    if (i + 1) % 10 == 0:
        print(f"  built {i+1}/{N_SPLITS}")


print("\nBaseline once per split ...")
base = []
for r, (Xd_b, Xh_b, yd, yh, *_) in enumerate(prepped):
    base.append(run(Xd_b, yd, Xh_b, yh, r))
base = np.array(base)
b_raw, b_dm = base[:, 0], base[:, 1]
print(f"baseline raw {b_raw.mean():+.3f} ± {b_raw.std(ddof=1)/np.sqrt(N_SPLITS):.3f}"
      f"  demeaned {b_dm.mean():+.3f} ± {b_dm.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")

print()
for cfg in CONFIGS:
    K, ck, bs, pw = cfg["K"], cfg["conf_k"], cfg["bar_start"], cfg["power"]
    scores = []
    for r, (Xd_b, Xh_b, yd, yh, dev_s, hold_s, dev_h_tag, hold_h_tag, cache_kc) in enumerate(prepped):
        dev_ic, hold_ic = cache_kc[(K, ck)]
        Xd = Xd_b.copy(); Xh = Xh_b.copy()
        Xd["event_impact_conf"] = aggregate(dev_h_tag, dev_ic, dev_s, bs, pw)
        Xh["event_impact_conf"] = aggregate(hold_h_tag, hold_ic, hold_s, bs, pw)
        scores.append(run(Xd, yd, Xh, yh, r))
    scores = np.array(scores)
    d_raw = scores[:, 0] - b_raw
    d_dm = scores[:, 1] - b_dm
    se_r = d_raw.std(ddof=1) / np.sqrt(N_SPLITS)
    se_d = d_dm.std(ddof=1) / np.sqrt(N_SPLITS)
    tr_ = d_raw.mean() / se_r if se_r > 0 else 0.0
    td_ = d_dm.mean() / se_d if se_d > 0 else 0.0
    print(f"K={K:2d} ck={ck:4.1f} bs={bs:2d} pw={pw:.1f}  "
          f"Δraw={d_raw.mean():+.3f}±{se_r:.3f}(t={tr_:+.2f})  "
          f"Δdm={d_dm.mean():+.3f}±{se_d:.3f}(t={td_:+.2f})")
