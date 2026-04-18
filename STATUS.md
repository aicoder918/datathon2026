# Status — Zurich Datathon 2026 (as of 2026-04-19)

## Current champion: `ridge_all_strong.csv` → LB **2.667**

Pure-long baseline (all positions = 1.0) scores **2.2**, so the ridge model is
contributing **+0.47 of real feature-alpha** on top of drift. The tilt is
genuine, not cosmetic.

## Model (what's winning right now)

- **Model**: strong-α Ridge regression (sklearn `Ridge`, α ≈ 5000–20000)
- **Features**: same 40 columns used by the production CatBoost pipeline
  (18 bar + 9 headline-sentiment aggregates + 12 OOF event-impact cols + 1 vol).
  See [FEATURES.md](FEATURES.md).
- **Prediction → position shaping**: `pred / vol`, threshold bottom 35% by
  |pred| to zero, normalize so mean-abs = 1, blend with constant-long at α=0.5,
  floor shorts at 0.30. Identical pipeline to the previous CatBoost champion.
- **Script**: built by the ChatGPT-generated `submissions/chatgpt/ridge_all_strong.csv`;
  our reproducible analog is [models/ridge_submission.py](models/ridge_submission.py)
  and [models/ridge_alpha_sweep.py](models/ridge_alpha_sweep.py).

## Leaderboard — full session history

### Ridge family (BREAKTHROUGH: 2.586 → 2.667)

| Submission | LB | Notes |
|---|---|---|
| **ridge_all_strong** | **2.667** | champion — strong-α Ridge, 40 base features |
| ridge_alpha_3000 | 2.660 | Ridge α=3000 |
| ridge_alpha_1500 | 2.653 | Ridge α=1500 |
| ridge_all | 2.653 | Ridge CV-picked α (~1000) |
| catboost_bars_ridge | 2.646 | 50/50 cat+ridge blend from `ridge_submission.py` |
| ridge_all_plus_catblend_50 | 2.649 | 50/50 cat+ridge blend |
| ridge_poly_a30000 | 2.645 | +780 pairwise interactions, α=30000 |
| ridge_all_killshorts | 2.630 | killshorts HURTS ridge (−0.023) |
| ridge_bars_headline_event_blend | 2.600 | subset blend |
| ridge_bars_event_blend | 2.587 | subset blend |
| ridge_poly_a3000 | 2.550 | interactions hurt more at low α |
| ridge_bars_only | 2.410 | bars alone weak |
| ridge_bootstrap_B200_a10000 | 2.622 | bootstrap averaging **HURT** (−0.045) |
| ridge_finbert_a10000 | **2.327** | finBERT [CLS] embeddings catastrophic (−0.340) |
| ridge_finbert_a30000 | 2.322 | strong α can't rescue train-noisy dims |

### CatBoost family (plateaued at 2.583–2.586)

| Submission | LB | Notes |
|---|---|---|
| catboost_bars_seed50_killshorts_blend | 2.586 | prior champion |
| catboost_bars_seed50_killshorts_blend_k55 | 2.586 | |
| catboost_bars_seed50_killshorts_blend_k45 | 2.586 | |
| catboost_bars_seed5 | 2.584 | 5-seed CatBoost ensemble |
| catboost_bars_seed5_hlv2 | 2.583 | same-count feature swap (≈flat) |
| catboost_bars_bootstrap30 | 2.585 | |
| catboost_bars_seed50_kelly | 2.553 | Kelly μ/(vol²+σ²) sizing failed |
| catboost_bars_seed5_sentv2 | 2.530 | "better" sent features failed |
| catboost_bars_pseudolabel_q10_w02 | 2.527 | pseudolabeling failed |
| pure_long (position=1.0) | 2.200 | drift-only baseline |

### What transferred across experiments

Only **variance-reduction** levers that keep the coefficient vector / feature
set *unchanged*:

- CatBoost seed ensembling: 2.570 → 2.583 (+0.013)
- Short-floor raise (0.30 → 0.40 on CatBoost): +0.006

Everything that **rotates** the coefficient vector or adds new signal-carrying
dimensions has failed LB, even when paired-diff loved it locally.

## Refined 8-strike / transfer rule (what we've learned)

Paired-diff on 120 splits drawn from the 1000 train rows is **not a trustworthy
predictor of LB**. The 20k test distribution is systematically different from
the 1k train distribution at paired-diff's resolution. Rules that have held:

1. **Variance-reduction that leaves the feature set unchanged transfers.**
   Seed ensembling, short-floor adjustments.
2. **Anything that rotates the model's direction fails.** Feature additions,
   feature replacements, interaction expansions, embedding augmentation,
   different targets, different losses.
3. **Bootstrap averaging on ridge fails even though it's structurally analogous
   to seed averaging on CatBoost.** Ridge on 1k rows with strong α is already
   low-variance; bootstrap mostly compressed real tail signal (min 0.30→0.50,
   max 2.89→2.35) and lost −0.045 LB. Ridge's shorts are real signal, not noise
   — consistent with `ridge_all_killshorts` (−0.023) and opposite of CatBoost
   where killshorts helped.
4. **HP search via paired-diff overfits even at t=+7.** Optuna found Δ+0.305
   (both halves +) → LB got worse.

## Why ridge beat CatBoost (what we think is happening)

CatBoost can represent millions of tree-interaction terms on 1k samples —
exactly the capacity that fits 1k-sample idiosyncrasies and fails to transfer
to 20k test. Strong-α Ridge on 40 features has mechanically-bounded capacity:
it cannot represent interactions. Whatever it learns must be surface-level
linear trends, and those happen to be the parts that generalize.

## Strikes (LB failures with strong local signal)

1. Optuna HP search — Δ+0.305 (t+7.11) → LB 2.51–2.55 vs 2.57
2. `yvol` target transform
3. Decay sentiment features — Δ+0.103 → LB 2.53
4. Next-momentum / MLM pretraining
5. Slim-column removal
6. `sent_v2` principled sentiment replacement — Δ+0.120 → LB 2.53
7. Cross-family blend (LightGBM) — LB −0.17
8. Kelly μ/(vol²+σ²) sizing — Δ+0.026 → LB 2.553
9. FinBERT [CLS] embeddings — LB 2.33 (biggest miss, −0.34)
10. Bootstrap ridge averaging — LB 2.62 (−0.045)

## Experiments that confirmed negative space (not strikes — purposeful tests)

- **Big ablation** (no_hl, no_evt, mini3, core7): feature set is at capacity,
  not bloated. Every group carries signal.
- **Cross-family LGB blending**: LGB genuinely weaker on this feature set;
  50/50 blend drags down by half the penalty.
- **Direct Sharpe MLP**: Sharpe gradients too noisy on 800 sessions for
  differentiable-Sharpe training to converge.

## Open levers (not yet tried; ordered by expected value)

1. **Shape knobs on ridge**: lower `short_floor` below 0.30 (give ridge's
   legitimate shorts more room) or lower `shrink_alpha` below 0.5 (more of the
   raw ridge tilt). Script ready at
   [models/ridge_shape_submit.py](models/ridge_shape_submit.py).
2. **α extension**: α ∈ {20000, 50000}. Monotone trend says higher wins; we
   don't know the peak.
3. **Ridge + ridge blend**: 50/50 blend of α=3000 and α=strong predictions
   — sample multiple points on the regularization path without pulling in
   CatBoost errors.
4. **PLS regression** (`pls_sweep.py`, `pls_c{3,5}.csv` exist but unsubmitted).
   PLS picks y-aligned components — different regularization mechanism than L2.
5. **Quantile-transformed features**: rank-transform each column before ridge.
   Cheap experiment against ridge's outlier sensitivity.

## Dead-end levers (do not retry)

- Feature additions to ridge (finBERT, poly interactions) — train-specific
  correlations even under strong L2
- Feature replacements on CatBoost (decay_sent, sent_v2, hlv2)
- Bootstrap resampling ridge
- Direct-Sharpe MLP / sequence models on the bar data
- CatBoost HP search by paired-diff
- Kelly μ/σ² sizing with model-predicted σ²
- Pseudolabeling, LightGBM blending

## Files

- [models/features.py](models/features.py) — 40-col feature pipeline (bars +
  headlines + event-impact)
- [models/ridge_alpha_sweep.py](models/ridge_alpha_sweep.py) — fixed-α ridge
  submissions
- [models/ridge_submission.py](models/ridge_submission.py) — our reproducible
  ridge + blend pipeline
- [models/ridge_shape_submit.py](models/ridge_shape_submit.py) — next to try
  (lower short_floor)
- [models/catboost_bars.py](models/catboost_bars.py) — previous CatBoost champion
- `submissions/chatgpt/` — ChatGPT-generated submissions, incl. champion
  (gitignored; lives on disk)
