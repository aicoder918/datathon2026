"""Grid search over event_impact_conf hyperparameters.

For each (K, conf_k) we fit pair/tid stats OOF on dev and naively on holdout,
then cheaply sweep (bar_start, recency_power) by re-aggregating contributions.

Paired-diff vs the 34-col baseline across N_SPLITS random splits.
"""
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

N_SPLITS = 25
HOLDOUT_FRAC = 0.2
ITERS = 52
BEST_KIND = "thresholded_inv_vol"

K_GRID = [3, 5, 10]
CONFK_GRID = [3.0, 10.0, 30.0]
BAR_START_GRID = [30, 35, 40]
POWER_GRID = [0.0, 1.0, 2.0]
MAX_BAR = 50

X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]


def contribs_from_stats(h_tagged: pd.DataFrame, tid_stats, pair_stats) -> np.ndarray:
    """Returns a per-headline `infl*conf` (ignoring recency) aligned to h_tagged rows."""
    out = np.zeros(len(h_tagged), dtype=float)
    tids = h_tagged["_tid"].to_numpy(int)
    secs = h_tagged["_sec"].to_numpy()
    for i, (t, s) in enumerate(zip(tids, secs)):
        stat = pair_stats.get((int(t), str(s))) or tid_stats.get(int(t))
        if stat is None:
            continue
        infl, conf = stat
        out[i] = infl * conf
    return out


def aggregate(h_tagged: pd.DataFrame, ic: np.ndarray, sessions: np.ndarray,
              bar_start: int, power: float) -> np.ndarray:
    """Sum infl*conf*recency^p per session."""
    span = float(MAX_BAR - bar_start)
    bars = h_tagged["bar_ix"].to_numpy(float)
    mask = (bars >= bar_start) & (bars < MAX_BAR)
    r = np.clip((bars - (bar_start - 1)) / span, 0.0, 1.0)
    vals = ic * (r ** power) * mask
    df = pd.DataFrame({"session": h_tagged["session"].values, "v": vals})
    agg = df.groupby("session")["v"].sum()
    out = pd.Series(0.0, index=sessions)
    out.loc[agg.index.intersection(sessions)] = agg.reindex(
        agg.index.intersection(sessions)).values
    return out.values


print(f"Pre-building baseline + conf stats across {N_SPLITS} splits ...")
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

    # per-K/conf_k: build OOF dev ic and holdout ic per headline
    dev_h_tag = _attach_tid_sec(dev_h)
    dev_h_tag = dev_h_tag[dev_h_tag["_tid"] >= 0].reset_index(drop=True)
    hold_h_tag = _attach_tid_sec(hold_h)
    hold_h_tag = hold_h_tag[hold_h_tag["_tid"] >= 0].reset_index(drop=True)

    per_kc = {}
    for K in K_GRID:
        for ck in CONFK_GRID:
            # OOF dev ic
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            dev_ic = np.zeros(len(dev_h_tag), dtype=float)
            sess_arr = dev_h_tag["session"].to_numpy()
            session_to_idx = {s: np.where(sess_arr == s)[0] for s in dev_s}
            for tr_idx, va_idx in kf.split(dev_s):
                tr_s = dev_s[tr_idx]
                va_s = dev_s[va_idx]
                h_tr = dev_h[dev_h["session"].isin(tr_s)]
                b_tr = dev_b[dev_b["session"].isin(tr_s)]
                t_stats, p_stats = fit_template_impacts_confidence(h_tr, b_tr, K=K, conf_k=ck)
                va_row_idx = np.concatenate([session_to_idx[s] for s in va_s if s in session_to_idx])
                if len(va_row_idx):
                    dev_ic[va_row_idx] = contribs_from_stats(
                        dev_h_tag.iloc[va_row_idx], t_stats, p_stats)
            # holdout: fit on full dev
            t_stats, p_stats = fit_template_impacts_confidence(dev_h, dev_b, K=K, conf_k=ck)
            hold_ic = contribs_from_stats(hold_h_tag, t_stats, p_stats)
            per_kc[(K, ck)] = (dev_ic, hold_ic)

    prepped.append((Xd_base, Xh_base, yd, yh, dev_s, hold_s,
                    dev_h_tag, hold_h_tag, per_kc))
    if (i + 1) % 5 == 0:
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


print("\nEvaluating baseline once per split ...")
base_scores = []
for r, (Xd_b, Xh_b, yd, yh, *_) in enumerate(prepped):
    base_scores.append(run(Xd_b, yd, Xh_b, yh, r))
base_scores = np.array(base_scores)
b_raw, b_dm = base_scores[:, 0], base_scores[:, 1]
print(f"baseline raw {b_raw.mean():+.3f} ± {b_raw.std(ddof=1)/np.sqrt(N_SPLITS):.3f}"
      f"  |  demeaned {b_dm.mean():+.3f} ± {b_dm.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")

print(f"\nGrid: K={K_GRID} × conf_k={CONFK_GRID} × bar_start={BAR_START_GRID} × power={POWER_GRID}")
print(f"Total configs: {len(K_GRID)*len(CONFK_GRID)*len(BAR_START_GRID)*len(POWER_GRID)}\n")

rows = []
for K in K_GRID:
    for ck in CONFK_GRID:
        for bs in BAR_START_GRID:
            for pw in POWER_GRID:
                conf_scores = []
                for r, (Xd_b, Xh_b, yd, yh, dev_s, hold_s,
                        dev_h_tag, hold_h_tag, per_kc) in enumerate(prepped):
                    dev_ic, hold_ic = per_kc[(K, ck)]
                    dev_col = aggregate(dev_h_tag, dev_ic, dev_s, bs, pw)
                    hold_col = aggregate(hold_h_tag, hold_ic, hold_s, bs, pw)
                    Xd = Xd_b.copy()
                    Xh = Xh_b.copy()
                    Xd["event_impact_conf"] = dev_col
                    Xh["event_impact_conf"] = hold_col
                    conf_scores.append(run(Xd, yd, Xh, yh, r))
                conf_scores = np.array(conf_scores)
                d_raw = conf_scores[:, 0] - b_raw
                d_dm = conf_scores[:, 1] - b_dm
                se_raw = d_raw.std(ddof=1) / np.sqrt(N_SPLITS)
                se_dm = d_dm.std(ddof=1) / np.sqrt(N_SPLITS)
                t_raw = d_raw.mean() / se_raw if se_raw > 0 else 0.0
                t_dm = d_dm.mean() / se_dm if se_dm > 0 else 0.0
                rows.append((K, ck, bs, pw, d_raw.mean(), se_raw, t_raw,
                             d_dm.mean(), se_dm, t_dm))
                print(f"K={K:2d} ck={ck:4.1f} bs={bs:2d} pw={pw:.1f}  "
                      f"Δraw={d_raw.mean():+.3f}±{se_raw:.3f}(t={t_raw:+.2f})  "
                      f"Δdm={d_dm.mean():+.3f}±{se_dm:.3f}(t={t_dm:+.2f})")

rows.sort(key=lambda x: -x[6])  # by t_raw descending
print("\nTop 10 by Δraw t-stat:")
for K, ck, bs, pw, dr, sr, tr_, dd, sd, td in rows[:10]:
    print(f"  K={K:2d} ck={ck:4.1f} bs={bs:2d} pw={pw:.1f}  "
          f"Δraw={dr:+.3f}±{sr:.3f}(t={tr_:+.2f})  Δdm={dd:+.3f}(t={td:+.2f})")
