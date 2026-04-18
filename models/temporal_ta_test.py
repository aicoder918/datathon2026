"""Paired-diff test: temporal-interaction + TA indicators + Huber loss.

Baseline  = production 39 cols, MAE, seed-5 ensemble.
+temporal = 10 price-response-to-headline features inside each session.
+ta       = 8 technical indicators (RSI, MACD×3, Bollinger×2, ATR, Garman-Klass).
+huber    = same features as baseline but CatBoost Huber loss.
+all      = temporal + ta (no huber, to isolate feature contribution).

Seed-5 ensemble per variant. 120 splits. Same per-split event-feature prep.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from features import (
    SEED, load_train_base, SENT_MAP,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_sector_multi, build_event_features_sector_multi,
    build_event_features_sector_oof,
    sharpe, shape_positions,
)

N_SPLITS = 120
HOLDOUT_FRAC = 0.2
ITERS = 52
N_SEEDS = 5
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2


# --- temporal features (price response to headlines, inside-session) ---
def build_temporal_features(headlines_df: pd.DataFrame, bars_df: pd.DataFrame,
                             all_sessions: np.ndarray) -> pd.DataFrame:
    """10 temporal-interaction features per session."""
    cols = [
        "ret_after_hl_mean", "ret_after_hl_std", "ret_before_hl_mean",
        "vol_hl_bars", "vol_nonhl_bars",
        "hl_ret_corr", "max_ret_post_hl", "min_ret_post_hl",
        "hl_cluster_count", "n_hl_bars",
    ]
    out = pd.DataFrame(0.0, index=all_sessions, columns=cols)
    out.index.name = "session"

    # build per-session bar returns
    b = bars_df[bars_df["bar_ix"] < 50].sort_values(["session", "bar_ix"]).copy()
    b["ret"] = b.groupby("session")["close"].pct_change().fillna(0.0)

    # sentiment lookup for headline amplitude proxy
    hdf = headlines_df.merge(SENT_MAP, left_on="headline", right_index=True, how="left")
    hdf["amp"] = hdf["signed"].abs().fillna(0.0)

    # group for efficiency
    bars_by_s = {s: sub for s, sub in b.groupby("session")}
    hdls_by_s = {s: sub for s, sub in hdf.groupby("session")}

    for s in all_sessions:
        bs = bars_by_s.get(s)
        if bs is None or len(bs) < 2:
            continue
        ret = bs["ret"].to_numpy(dtype=float)
        bar_ix = bs["bar_ix"].to_numpy(dtype=int)
        ret_by_ix = np.zeros(50, dtype=float)
        ret_by_ix[bar_ix[bar_ix < 50]] = ret[bar_ix < 50]

        hh = hdls_by_s.get(s)
        if hh is None or len(hh) == 0:
            # no headlines: vol_nonhl = full vol, everything else 0
            out.at[s, "vol_nonhl_bars"] = float(ret_by_ix.std())
            continue

        h_bars = hh["bar_ix"].to_numpy(int)
        h_bars = h_bars[(h_bars >= 0) & (h_bars < 50)]
        if len(h_bars) == 0:
            out.at[s, "vol_nonhl_bars"] = float(ret_by_ix.std())
            continue
        h_amp = hh["amp"].to_numpy(float)[:len(h_bars)]
        uniq_h_bars = np.unique(h_bars)

        # returns immediately after headlines (bar_ix + 1)
        post = uniq_h_bars + 1
        post = post[post < 50]
        if len(post) > 0:
            out.at[s, "ret_after_hl_mean"] = float(ret_by_ix[post].mean())
            out.at[s, "ret_after_hl_std"] = float(ret_by_ix[post].std()) if len(post) > 1 else 0.0
        # returns immediately before
        pre = uniq_h_bars - 1
        pre = pre[pre >= 0]
        if len(pre) > 0:
            out.at[s, "ret_before_hl_mean"] = float(ret_by_ix[pre].mean())

        # vol on headline vs non-headline bars
        hl_mask = np.zeros(50, bool); hl_mask[uniq_h_bars] = True
        if hl_mask.sum() > 1:
            out.at[s, "vol_hl_bars"] = float(ret_by_ix[hl_mask].std())
        nm = ~hl_mask
        if nm.sum() > 1:
            out.at[s, "vol_nonhl_bars"] = float(ret_by_ix[nm].std())

        # corr(amplitude, candle return) — matched pairs (h_bar, amp)
        h_bars_valid = h_bars < 50
        if h_bars_valid.sum() > 2:
            h_ret = ret_by_ix[h_bars[h_bars_valid]]
            h_amp_v = h_amp[h_bars_valid]
            if h_ret.std() > 1e-12 and h_amp_v.std() > 1e-12:
                out.at[s, "hl_ret_corr"] = float(np.corrcoef(h_ret, h_amp_v)[0, 1])

        # max/min ret in the 3 bars after any headline
        post3 = np.unique(np.clip(np.concatenate([uniq_h_bars + k for k in (1, 2, 3)]), 0, 49))
        if len(post3) > 0:
            out.at[s, "max_ret_post_hl"] = float(ret_by_ix[post3].max())
            out.at[s, "min_ret_post_hl"] = float(ret_by_ix[post3].min())

        # cluster count: # of 3-bar windows with ≥ 2 headlines
        if len(uniq_h_bars) >= 2:
            cluster = 0
            sb = np.sort(uniq_h_bars)
            for i in range(len(sb)):
                j = i + 1
                while j < len(sb) and sb[j] - sb[i] < 3:
                    j += 1
                if j - i >= 2:
                    cluster += 1
            out.at[s, "hl_cluster_count"] = float(cluster)

        out.at[s, "n_hl_bars"] = float(len(uniq_h_bars))

    return out.sort_index()


# --- TA indicators (OHLC-only, last-50 bars) ---
def build_ta_features(bars_df: pd.DataFrame, all_sessions: np.ndarray) -> pd.DataFrame:
    cols = ["rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_width", "bb_pct", "atr_14", "gk_vol"]
    out = pd.DataFrame(0.0, index=all_sessions, columns=cols)
    out.index.name = "session"

    b = bars_df[bars_df["bar_ix"] < 50].sort_values(["session", "bar_ix"])
    for s, sub in b.groupby("session"):
        c = sub["close"].to_numpy(float)
        o = sub["open"].to_numpy(float)
        h = sub["high"].to_numpy(float)
        lo = sub["low"].to_numpy(float)
        if len(c) < 20:
            continue

        # RSI-14
        d = np.diff(c)
        up = np.clip(d, 0, None); dn = -np.clip(d, None, 0)
        if len(up) >= 14:
            au = up[-14:].mean(); ad = dn[-14:].mean()
            rs = au / ad if ad > 1e-12 else 0.0
            rsi = 100 - 100 / (1 + rs) if ad > 1e-12 else 100.0
            out.at[s, "rsi_14"] = float(rsi)

        # MACD: EMA(12) - EMA(26), signal = EMA(9) of macd
        def ema(x, span):
            a = 2.0 / (span + 1)
            e = np.zeros_like(x, dtype=float)
            e[0] = x[0]
            for i in range(1, len(x)):
                e[i] = a * x[i] + (1 - a) * e[i - 1]
            return e
        if len(c) >= 26:
            macd_line = ema(c, 12) - ema(c, 26)
            sig = ema(macd_line, 9)
            out.at[s, "macd"] = float(macd_line[-1])
            out.at[s, "macd_signal"] = float(sig[-1])
            out.at[s, "macd_hist"] = float(macd_line[-1] - sig[-1])

        # Bollinger 20
        if len(c) >= 20:
            win = c[-20:]
            m = win.mean(); sd = win.std()
            upper = m + 2 * sd; lower = m - 2 * sd
            out.at[s, "bb_width"] = float(upper - lower)
            if upper > lower:
                out.at[s, "bb_pct"] = float((c[-1] - lower) / (upper - lower))

        # ATR-14 (TR = max(H-L, |H-Cprev|, |L-Cprev|))
        if len(c) >= 15:
            hl = h[1:] - lo[1:]
            hc = np.abs(h[1:] - c[:-1])
            lc = np.abs(lo[1:] - c[:-1])
            tr = np.maximum.reduce([hl, hc, lc])
            out.at[s, "atr_14"] = float(tr[-14:].mean())

        # Garman-Klass: 0.5*log(H/L)^2 - (2*ln2 - 1)*log(C/O)^2, averaged
        ln2 = np.log(2.0)
        gk = 0.5 * np.log(np.maximum(h, 1e-12) / np.maximum(lo, 1e-12)) ** 2 \
             - (2 * ln2 - 1) * np.log(np.maximum(c, 1e-12) / np.maximum(o, 1e-12)) ** 2
        out.at[s, "gk_vol"] = float(gk.mean())

    return out.sort_index()


# --- prep splits ---
X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()

print("Pre-computing temporal + TA features once (session-local) ...")
temporal_all = build_temporal_features(headlines_train, bars_train, all_sessions)
ta_all = build_ta_features(bars_train, all_sessions)
print(f"  temporal: {temporal_all.shape}, TA: {ta_all.shape}")

n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

print(f"Pre-building per-split event features across {N_SPLITS} splits ...")
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
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]
    prepped.append((Xd, Xh, yd, yh))
    if (i + 1) % 20 == 0:
        print(f"  built {i+1}/{N_SPLITS}")


def cb_pred(Xd, yd, Xh, seed, loss="MAE"):
    m = CatBoostRegressor(iterations=ITERS, learning_rate=0.03, depth=5,
                         loss_function=loss, random_seed=seed, verbose=False)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def score_variant(add_temporal, add_ta, loss="MAE"):
    scores = []
    for r, (Xd, Xh, yd, yh) in enumerate(prepped):
        Xd_v = Xd; Xh_v = Xh
        if add_temporal:
            Xd_v = Xd_v.join(temporal_all.loc[Xd_v.index])
            Xh_v = Xh_v.join(temporal_all.loc[Xh_v.index])
        if add_ta:
            Xd_v = Xd_v.join(ta_all.loc[Xd_v.index])
            Xh_v = Xh_v.join(ta_all.loc[Xh_v.index])
        vh = np.asarray(Xh_v["vol"].values, dtype=float)
        y_arr = np.asarray(yh.values, dtype=float)
        preds = [cb_pred(Xd_v, yd, Xh_v, seed=SEED + r * 997 + k, loss=loss)
                 for k in range(N_SEEDS)]
        pr = np.mean(preds, axis=0)
        scores.append(sharpe(shape_positions(pr, vh, "thresholded_inv_vol",
                                             threshold_q=THRESHOLD_Q), y_arr))
        if (r + 1) % 20 == 0:
            print(f"  scored {r+1}/{N_SPLITS}")
    return np.asarray(scores)


def report(label, s, base):
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean()/se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean()/seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean()/seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "
    return (f"{label:<14} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
            f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")


print("\nScoring baseline (seed5 MAE) ...")
base = score_variant(False, False, loss="MAE")
print(f"baseline raw {base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")

print("\nScoring +temporal (seed5 MAE) ...")
tmp = score_variant(True, False, loss="MAE")
print(report("+temporal", tmp, base))

print("\nScoring +ta (seed5 MAE) ...")
ta = score_variant(False, True, loss="MAE")
print(report("+ta", ta, base))

print("\nScoring +all = +temporal+ta (seed5 MAE) ...")
both = score_variant(True, True, loss="MAE")
print(report("+temporal+ta", both, base))

print("\nScoring +huber (seed5 Huber:delta=0.002) ...")
hub = score_variant(False, False, loss="Huber:delta=0.002")
print(report("+huber", hub, base))

print("\nReliability filter — both halves Δ>0:")
for label, s in [("+temporal", tmp), ("+ta", ta), ("+temporal+ta", both), ("+huber", hub)]:
    d = s - base
    dA, dB = d[:HALF], d[HALF:]
    if dA.mean() > 0 and dB.mean() > 0:
        se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean()/se if se > 0 else 0.0
        print(f"  {label:<14} Δ={d.mean():+.3f}(t={t:+.2f})  "
              f"A:{dA.mean():+.3f}  B:{dB.mean():+.3f}")
