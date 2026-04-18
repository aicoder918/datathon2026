# Feature Inventory — Production Model (39 columns)

Defined in [models/features.py](models/features.py). Consumed by `load_train()` → CatBoost.

## 1. Bar features (18) — `build_bar_features`
Computed from the 50 seen bars per session.

| Column | Definition |
|---|---|
| `close_first` / `close_last` | First and last close in bars 0–49 |
| `seen_ret` | `close_last / close_first - 1` |
| `mom_1`, `mom_3`, `mom_5`, `mom_10` | Sum of last N per-bar returns |
| `vol` | Std of per-bar returns over all 50 bars |
| `vol_recent` | Std of per-bar returns over last 10 bars |
| `max_high`, `min_low` | Session extremes |
| `dist_to_high` | `max_high / close_last - 1` |
| `dist_to_low` | `min_low / close_last - 1` |
| `body_mean`, `range_mean` | Mean of `|close - open|` and `high - low` |
| `close_pos_mean`, `close_pos_last` | Mean / last of `(close - low) / (high - low)` |
| `max_drawdown` | Min of `close / cummax_close - 1` |
| `ret_skew` | Skewness of per-bar returns |

## 2. Headline / sentiment features (9) — `build_headline_features`
From `headlines_seen_train.parquet` joined with FinBERT sentiment lookup.

| Column | Definition |
|---|---|
| `hl_n`, `hl_n_recent` | Count of headlines in bars 0–49 / 40–49 |
| `hl_last_bar`, `hl_mean_bar` | Timing stats (bar_ix) |
| `hl_net_sent`, `hl_net_sent_recent` | Sum of FinBERT signed scores (all / recent) |
| `hl_mean_sent` | Mean FinBERT signed score |
| `hl_n_pos`, `hl_n_neg` | Count of positive / negative FinBERT labels |

`signed = +score` if label=positive, `-score` if negative, `0` if neutral.

## 3. Event-impact features (12) — OOF mean encoding
Per-template and per-(template, sector) forward-return lookups. Each template's
"impact" = shrunk mean forward-K-bar return across headlines matching that
template (fixed-prior shrinkage, `prior_n=30`, toward global mean).

Built with **out-of-fold 5-fold CV** on train (each session's features use
impacts fit on sessions in the other 4 folds) to prevent target leakage. Test
features use impacts fit on all training data.

For each horizon `K ∈ {3, 5, 10}`:
- `event_impact_k{K}` — sum of tid-only impacts across session's headlines
- `event_impact_recent_k{K}` — same, weighted by `exp(-(49 - bar_ix) / 10)`

And hierarchical (tid, sector) impacts (sector-level shrunk toward tid-level):
- `event_impact_sec_k{K}`
- `event_impact_sec_recent_k{K}`

→ 6 tid cols + 6 (tid, sector) cols = 12.

## Target
`y = close[99] / close[49] - 1` — the forward return from bar 49 (trade point)
to bar 99 (unwind point).

## Prediction → position shaping
CatBoost (depth=5, lr=0.03, MAE loss, ~58 iters CV-picked) outputs raw `pred`.
Then `shape_positions(pred, vol, "thresholded_inv_vol", threshold_q=0.35)`:
1. Zero preds with `|pred|` below the 35th percentile
2. Divide remaining preds by `vol`
3. Normalize so `mean(|pos|) = 1`
4. Blend with constant-long: `0.5 * scaled + 0.5 * 1.0`
5. Floor shorts at 0.3

The drift is positive and unconditional, so the constant-long blend and short
floor dominate; the model's job is to modulate size, not to flip sign often.

## Leaderboard results (current session)
- **2.583** — seed-5 CatBoost ensemble (validated LB-positive lever)
- 2.576 — kill-shorts variant (all negative preds → 0 before shaping)
- 2.570 — baseline single-seed CatBoost
