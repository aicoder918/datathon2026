"""Paired-diff: Kelly-style position sizing. Test whether replacing
`thresholded_inv_vol` shaping with a principled μ/σ² sizing improves Sharpe.

Sources of σ tested:
  - bar vol (the same vol we already use for inv_vol shaping, now squared)
  - model uncertainty from CatBoost RMSEWithUncertainty (virtual ensembles)
  - combined vol² + σ_model²

Variants (all use the SAME seeded CatBoost predictions as baseline; only the
shape function changes):
  baseline         — current: thresholded_inv_vol (q=0.35) with finalize
  kelly_vol        — μ / vol², thresholded, finalized
  kelly_modelvar   — μ / σ_model², thresholded, finalized
  kelly_combined   — μ / (vol² + σ_model²), thresholded, finalized
  kelly_pure       — μ / σ² with NO threshold, NO finalize (raw Kelly)

Bar for LB submission: Δ > 0 on both halves, Δ modest (<0.10), AND must survive
dropping threshold (i.e. kelly_vol beats baseline AND kelly_pure not far
behind) — if only thresholded version wins, the "gain" is just threshold
re-tuning which is an HP we already know doesn't transfer.
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
    sharpe, shape_positions, finalize,
    SHRINK_ALPHA, SHORT_FLOOR,
)

N_SPLITS = 60
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
N_SEEDS = 3

CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)
# Uncertainty model: RMSEWithUncertainty trained separately so baseline μ
# predictions are identical to our existing pipeline.
CB_UNC_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
                 loss_function="RMSEWithUncertainty",
                 posterior_sampling=True, verbose=False)


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def cb_unc_pred(Xd, yd, Xh, seed):
    """Train an RMSEWithUncertainty model and return (mu, var)."""
    m = CatBoostRegressor(**CB_UNC_KW, random_seed=seed)
    m.fit(Xd, yd)
    p = m.virtual_ensembles_predict(
        Xh, prediction_type="TotalUncertainty",
        virtual_ensembles_count=10, verbose=False,
    )
    mu = np.asarray(p[:, 0], dtype=float)
    # p[:, 1] is KnowledgeUncertainty, p[:, 2] is DataUncertainty (aleatoric)
    # For position sizing, we want total predictive variance
    var = np.asarray(p[:, 1] + p[:, 2], dtype=float)
    return mu, var


def apply_threshold_finalize(pos: np.ndarray, raw_pred: np.ndarray,
                              threshold_q: float = THRESHOLD_Q) -> np.ndarray:
    """Threshold positions to zero where |raw_pred| < q-quantile, then finalize."""
    cutoff = np.quantile(np.abs(raw_pred), threshold_q)
    pos = pos.copy()
    pos[np.abs(raw_pred) < cutoff] = 0.0
    return finalize(pos)


def sharpe_score(pos, y):
    return sharpe(pos, y)


X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

VARIANTS = ["baseline", "kelly_vol", "kelly_modelvar", "kelly_combined", "kelly_pure"]
scores = {v: [] for v in VARIANTS}

print(f"Paired-diff: {N_SPLITS} splits × {N_SEEDS} seeds ...")
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
    Xd = X_base.loc[dev_s].join(dev_event).join(dev_sec)
    Xh = X_base.loc[hold_s].join(hold_event).join(hold_sec)
    yd = y_full.loc[dev_s].to_numpy(dtype=float)
    yh = y_full.loc[hold_s].to_numpy(dtype=float)
    vh = np.asarray(Xh["vol"].values, dtype=float)

    # μ predictions (seed-averaged, same as baseline)
    mus = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)

    # σ_model² from a single uncertainty model (don't need seed average for σ)
    _, var_model = cb_unc_pred(Xd, yd, Xh, seed=SEED + r * 997)
    var_model = np.clip(var_model, 1e-8, None)

    # Variance from bar vol (squared)
    var_vol = np.clip(vh ** 2, 1e-8, None)

    # baseline: thresholded_inv_vol + finalize
    pos_base = shape_positions(mus, vh, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos_base = finalize(pos_base)
    scores["baseline"].append(sharpe_score(pos_base, yh))

    # kelly_vol: μ / vol², then threshold + finalize
    pos = mus / var_vol
    pos = apply_threshold_finalize(pos, mus, THRESHOLD_Q)
    scores["kelly_vol"].append(sharpe_score(pos, yh))

    # kelly_modelvar: μ / σ_model², then threshold + finalize
    pos = mus / var_model
    pos = apply_threshold_finalize(pos, mus, THRESHOLD_Q)
    scores["kelly_modelvar"].append(sharpe_score(pos, yh))

    # kelly_combined: μ / (vol² + σ_model²)
    pos = mus / (var_vol + var_model)
    pos = apply_threshold_finalize(pos, mus, THRESHOLD_Q)
    scores["kelly_combined"].append(sharpe_score(pos, yh))

    # kelly_pure: μ / (vol² + σ_model²), no threshold, no finalize — raw Kelly
    pos = mus / (var_vol + var_model)
    scores["kelly_pure"].append(sharpe_score(pos, yh))

    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}  var_model mean={var_model.mean():.2e}  var_vol mean={var_vol.mean():.2e}")

base = np.asarray(scores["baseline"])
print(f"\nbaseline raw  mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
for v in VARIANTS[1:]:
    s = np.asarray(scores[v])
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else (
           "!" if dA.mean() < 0 and dB.mean() < 0 else " ")
    print(f"  {v:<17} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")
