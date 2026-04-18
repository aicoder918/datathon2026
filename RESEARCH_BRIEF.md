# Research Brief — Zurich Datathon 2026 Simulated Market Close Prediction

## Your task

Find published methods, competition writeups, or library techniques that could move our Sharpe score on the problem below. We are plateaued; we want **concrete methods we can implement in 1–2 days**, not broad surveys.

Return (in under 800 words):
1. 3–5 specific techniques worth trying, ranked by expected impact.
2. For each: one-line description, the paper/writeup URL, and the specific reason you think it applies to OUR problem.
3. Any "obvious lever" you think we're missing.

## The problem

**Synthetic single-stock session prediction.** Each session is an independent 100-bar simulated trading day for a single synthetic asset. We see bars 0-49 and headlines from bars 0-49. We predict one number per session: `target_position` ∈ ℝ. At bar 49 we trade that quantity; at bar 99 we unwind. Score = Sharpe ratio across sessions × 16:

```
pnl_i = target_position_i * (close[99] / close[49] - 1)
sharpe = mean(pnl) / std(pnl) * 16
```

1,000 training sessions (with full bars + headlines), 20,000 test sessions (seen half only). Each session has OHLC for every bar and 10–30 headlines distributed across bars 0-49.

Headlines are templated (~50 distinct templates) and reference fictional companies, sectors, and regions. Each session's asset is not explicitly named — headlines may or may not be relevant to the session's asset. Example templates:
- `<COMPANY> reports strong demand in <REGION>, raises outlook`
- `<COMPANY> sees N% drop in new customer orders this quarter`
- `<COMPANY> <ROLE> steps down unexpectedly citing <REASON>`
- `<COMPANY> secures $X contract with <PARTNER>`

Prices are normalized to start at 1.0. Headlines are pre-scored for sentiment with FinBERT.

## Current approach (2.57 LB)

- **Model**: CatBoost regressor, MAE loss, depth=5, lr=0.03, ~58 iters (CV-picked). Trained on 1,000 sessions, predicted on 20,000.
- **Features (39 cols)**:
  - 18 bar features from the 50 seen bars: close_first/last, seen_ret, mom_{1,3,5,10}, vol, vol_recent, max_high, min_low, dist_to_{high,low}, body_mean, range_mean, close_pos_{mean,last}, max_drawdown, ret_skew.
  - 9 headline/sentiment: hl_n, hl_n_recent, hl_last/mean_bar, hl_net_sent (+ recent), hl_mean_sent, hl_n_pos, hl_n_neg.
  - 12 event-impact: OOF mean-encoded forward return per template_index (×3 horizons k=3,5,10) and per (template_index, sector) pair (×3 horizons). Each has a "recent-weighted" variant.
- **Prediction → position shaping**: `pred / vol`, threshold bottom 35% by |pred| to zero, scale so mean-abs = 1, blend with constant-long at α=0.5, floor shorts at 0.3 (most positions end up long, asymmetric — the drift is positive).

## What we've already tried and that DID NOT MOVE LB

- **Dropping column groups / single-keep slim**: 120-split paired-diff, no reliable drop.
- **5 new hand-crafted features** (vol_ratio, parkinson_vol, hl_first_bar, hl_span, sent_abs_max): all negative or null.
- **Linear blend** with LightGBM and Ridge: LightGBM=2.11, Ridge=1.92 alone, so blending pulls CatBoost down. Null.
- **Optuna TPE** over CatBoost hyperparams (iterations, depth, lr, l2_leaf_reg, rsm, random_strength, border_count). Found a "winner" at **local Δ+0.305 (t=+7.11, both halves positive)**, but LB got **worse** (2.57 → 2.51–2.55). Classic paired-diff overfit — the 120 splits all draw from the same 1,000 training sessions.
- **Relevance filter** (build event_impact only from headlines about each session's most-mentioned company) + **numeric magnitude parsing** (N%, $X): both negative, both halves negative.
- **Shape-parameter sweep** (quantile threshold, vol exponent, shrink α, short floor): local +0.054 → LB +0.01. Already baked in.
- **Swapping back to older pre-merge-rules CSVs**: we haven't tested this but currently the new CSV consolidates 73 templates → 50 (statistically preferable).

## Why we think we're plateaued

- The pairable-diff local validation draws from the same 1,000 sessions the features are built from; hyperparam search overfits this.
- The mapping from (bars + headlines) → forward return appears to have a ceiling the hand-engineered features keep hitting. All reliable local wins fail to transfer.
- Our shaping is already tuned; the model's raw predictive content is what's bottlenecked.
- Most likely untouched lever: the **sequential structure of the 50 bars** (we only use aggregates like mom_N, vol, drawdown). A sequence model might extract path features that tree-based summaries miss.

## Specific questions for you

1. **Bar-sequence modeling for tabular downstream.** Techniques for using a small 1D-CNN, LSTM, Transformer, or TCN to embed a fixed-length OHLC sequence, then feed the embedding as features to a GBM. Competition writeups (Kaggle, Numerai) welcome.
2. **Sharpe-aware training losses / position outputs.** Papers or blog posts on training models directly against Sharpe (or CVaR, information ratio) rather than MSE/MAE-then-shape. Differentiable Sharpe approximations.
3. **Asymmetric position sizing / Kelly with drift.** When the unconditional drift is positive and short-side confidence is inherently lower, how do practitioners shape positions? Concrete formulas preferred.
4. **News-event impact estimation with template/sector hierarchies.** We use OOF hierarchical shrinkage (tid → tid+sector → global). Are there better estimators (empirical Bayes, PyMC / bambi, variance-weighted GLM) for ~50 templates × small-N groups?
5. **Handling train-distribution overfit when validation draws from the training pool.** Standard techniques for honest HP tuning when you cannot hold out from the real test distribution.

## Data & code access (for context, don't need to touch)

Code lives in `/Users/mgershman/Desktop/datathon/datathon2026/models/`. Key files: `features.py`, `catboost_bars.py`, `template_parser.py`. Raw data in `../data/*.parquet`.

## Non-goals

- We are NOT looking for general ML advice ("try more regularization", "add dropout").
- We are NOT looking for a full ML research survey — we want 3–5 directly-actionable pointers.
