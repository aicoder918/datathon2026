# Strategy Summary — HRT ETH Zurich Datathon 2026

---

## Final Scoring Formula

For each test session `i` ∈ {1000, …, 20999}:

```
y_i = [0.910 × (v2k3_i − 0.28) + 0.12 × stk_i] / (parkinson_vol_i / median(parkinson_vol))^1.2
```

where `target_position = y`.

- `v2k3` = base signal (session-level bar+headline model blended with per-headline interactions model)
- `stk`  = orthogonal meta-stack signal
- `parkinson_vol` = per-session Parkinson (high/low-range) realized volatility from bars 0–49
- Shift `−0.28`: calibration (reduces variance of PnL).
- Scaler exponent `p = 1.2`: Kelly-style risk budgeting.

---

## Model Inventory (every model that fed into the final submission)

### 1. `champ` — session-level ensemble (standalone LB 2.90305)

Inherited from the handoff. Stored as `submissions/submission_best_plus_ridge_top10_935_065.csv`.

**Structure**: `0.935 × seed20_x103_base + 0.065 × ridge_top10`

- `seed20_x103_base`: the pre-existing "best" CatBoost/ridge session-level blend. 40-feature session matrix (bar aggregates + headline aggregates + event-impact features), produced via `models/` pipeline (see `models/features.py`).
- `ridge_top10`: `RidgeCV` on the 10 highest |corr(x, y)| features (see `models/ridge_submit.py`).

**Contribution**: Foundational — all signals downstream ride on top of this. Without it, every HL blend collapses to ~2.58.

### 2. `hl_inter_v2_k3_ens` — per-headline ridge with interaction features

Three ridge models with shared features but different L2 strength:
`ridge_hl_interv2_k3_a3000.csv`, `..._a10000.csv`, `..._a30000.csv` — averaged 1/3 each.

**Training data**: 9,740 headlines from the 1,000 training sessions (`headlines_seen_train.parquet`).

**Features (996 sparse columns)**:
- 80 base per-headline features: `template_id` one-hot (32 IDs), `sector` one-hot (10), `region` one-hot (10), `bar_ix`, `log(1 + $ amount)`, percentage, FinBERT signed score, positive/negative label dummies, session cumulative return up to `bar_ix`, session rolling-10 vol up to `bar_ix`.
- 928 interaction features (added via `run_hl_inter_v2.py`):
  - `template_id × bar_bucket` (3 buckets: 0–15, 16–35, 36–49) — 96 cols
  - `template_id × sector` — 320 cols
  - `sector × bar_bucket` — 30 cols
  - `region × bar_bucket` — 33 cols

**Target**: forward 3-bar return `close[bar_ix + 3] / close[bar_ix] − 1`, capped at session end.

**Aggregation per session**: `sum( prediction × exp(−(49 − bar_ix) / 10) )` across the session's headlines.

**Position shaping of the standalone model**: `thresholded_inv_vol` with quantile 0.35 + shrink α=0.5 + short floor 0.30.

**Contribution**: +0.020 when blended at 80/20 into champ (2.90306 → 2.92344).

### 3. `stk` — session-level stacking meta-ridge

`stack_hl_session_ridge.csv` (see `run_hl_stack.py`).

**Inputs**: 10 aggregate features per session derived from the OOF predictions of the HL interaction model (`sum_pred_rec`, `mean_pred_rec`, `max_pred_rec`, `min_pred_rec`, `std_pred`, `count_hl`, `sum_pos`, `sum_neg`, `last_bar`, `mean_bar`).

**Target**: session-level `last_unseen / last_seen − 1` (i.e. the competition target).

**Model**: `RidgeCV(alphas=logspace(-2, 4, 13))`, chose α≈10,000.

**Role**: Blended in at `w = 0.12` as an orthogonal correction (corr with `v2k3` ≈ 0.24, the lowest of any candidate).

**Contribution**: Alone ~+0.0009 at blend time. Larger contribution appears once vol-scaling is applied: it increases effective leverage on low-vol high-confidence sessions.

### 4. Parkinson vol `σ_P`

Per-session realized vol from **bars 0–49 of the seen test bars** (public + private concatenated):

```
σ_P(i) = sqrt( mean( ln(high/low)^2 ) / (4 ln 2) ) over bar_ix ∈ [0, 49]
```

This is not a learned model — it's a closed-form volatility estimator, but it has the biggest single impact on the LB (see table below).

---

## Models trained but NOT in the final submission

Useful to log because they establish what families of signal exist but don't contribute:

| Model | File | Why it didn't end up in the final blend |
|---|---|---|
| Per-headline ridge (base 80 features, no interactions) | `ridge_hl_a{30..3000}.csv` | Superseded by the interactions model (+0.007 more per same 80/20 blend). |
| Per-headline ridge **v1 interactions** (tid×bar, tid×sec only, 930 cols) | `ridge_hl_inter_k3_a*.csv` | 99.97% corr with v2, slightly worse — v2 chosen. |
| CatBoost on 80 HL features | `catboost_hl_k5.csv` | 0.82 corr with ridge HL, net dilutes when averaged. |
| CatBoost (depth=6, 800 iter) on 930 HL interactions | `catboost_hl_inter_k3_d6.csv` | Only 0.52 corr with ridge, but weaker Sharpe — dilutes. |
| ElasticNet on HL interactions | `enet_hl_inter_k3_*.csv` | 0.85 corr with ridge, ~0.008 worse. |
| Logistic classifier on sign(fwd K=5) with interactions | `logreg_hl_k5_inter.csv` | 0.99 corr with ridge — redundant. |
| K=1/2/5/10/session-end targets | `ridge_hl_inter_k{1,2,5,10,end}_a*.csv` | K=3 strictly best. |
| Semi-supervised pseudo-labels | `ridge_hl_inter_ssl_a*.csv` | 0.94 corr with supervised — no new info. |
| "Rich meta-ridge" (40 session features + HL aggregates) | `ridge_rich_meta.csv` | Weak standalone (2.62), dilutes when blended. |
| Company-name one-hot | `ridge_hl_interco_k3_a*.csv` | 0.9992 corr with non-company interactions — redundant. |
| FinBERT CLS embeddings (SVD-32) as HL features | `ridge_hl_interfb_k3_a*.csv` | 0.90 corr with base HL, dilutes. |
| MLM bar embeddings ridge | `ridge_mlm_a*.csv` | Hurts when added as orthogonal branch. |
| Rank-transformed ridge on 40 session features | `ridge_rank_a*.csv` | Hurts. |
| Expanded per-bar ridge (50 per-bar returns + windowed stats) | `ridge_barex_a*.csv` | Hurts. |
| Session-level PLS (c=3/5) | `chatgpt/pls_c{3,5}.csv` | Hurts. |
| Garman-Klass vol (uses OC too) | `v110_gk_p08.csv` | 2.99848 — strictly worse than Parkinson 3.00027. |
| Rogers-Satchell vol | (built, not submitted) | Similar to GK. |
| Close-to-close std vol | `v109_p080.csv` | 2.97658 — Parkinson beats it by +0.024. |

---

## Impact Ranking (sorted by marginal LB gain)

| Rank | Change | Gain on public LB | Cumulative |
|---|---|---|---|
| 1 | **Parkinson vol replaces close-to-close std vol** | +0.01556 | to 2.99661 |
| 2 | **HL per-headline ridge (K=5) blended in at 87/13** | +0.00958 | to 2.91264 |
| 3 | **Exponent p on vol scaler tuned 0.80 → 1.2** | +0.00434 | to 3.00083 |
| 4 | **HL interactions ensemble (triple-α)** | +0.00412 | to 2.91913 |
| 5 | **K=3 horizon** (instead of K=5) | +0.00331 | to 2.92327 |
| 6 | **Template × bar / template × sector interactions** | +0.00398 | to 2.91515 |
| 7 | **Additive calibration shift −0.24** | +0.00286 | to 2.97944 |
| 8 | **Add `stk` stack meta-ridge at w=0.12** | +0.00069 | to 2.98014 |
| 9 | Final shift `−0.03` on y | +0.00091 | to 2.98105 |
| 10 | Deeper shift `−0.26 → −0.28` at Parkinson p=1.2 | +0.00067 | to 3.00095 |
| 11 | v2 interactions (+ sec×bar, reg×bar) over v1 alone | +0.00017 | marginal |

**The two dominant levers (in order):**
1. **Parkinson vol scaling (+0.0156)** — largest single LB move. Why: Parkinson has ~1/5 the sampling variance of close-to-close std for the same number of bars, so the per-session vol estimate is more accurate → better Kelly-style position downweighting.
2. **Per-headline ridge with interactions (+0.020 cumulative across the HL stack)** — the only genuinely new signal we built. Captures per-template / per-sector reaction patterns that session-level aggregates smooth away.

Every other step moves the LB by <0.005. Blending weight tuning, shift tuning, alpha ensembling and so on are each responsible for O(1e-3).

---

## Why the Parkinson-vol-scaling step worked so well

Sharpe = mean(PnL) / std(PnL). Even with a fixed-direction position, reducing position size in high-σ sessions shrinks var(PnL) faster than it shrinks E[PnL] whenever the model's expected return is not proportional to vol. Close-to-close vol needs ~50 bars of data to hit ±15% accuracy; Parkinson hits the same accuracy with ~5× fewer bars because it uses the full intra-bar range, not just close prices. On 50-bar windows Parkinson's standard error is ~3× lower. The scaler is therefore much closer to the "true" per-session vol, and the division shrinks noise more consistently.

The extra gain from `p = 1.1–1.2` (vs naive `p = 1.0`) pushes a bit past strict inverse-vol weighting — consistent with the signal having heavier-than-Gaussian residuals, so aggressive downweighting of the worst-vol tail pays off.

---

## Reproduction

```python
import pandas as pd, numpy as np
from pathlib import Path
R = Path("/Users/mgershman/Desktop/datathon/datathon2026")

# 1. Load base components
champ = pd.read_csv(R/"submissions/submission_best_plus_ridge_top10_935_065.csv"
                   ).sort_values("session").reset_index(drop=True)
v2 = sum(pd.read_csv(R/f"submissions/ridge_hl_interv2_k3_a{a}.csv"
                    ).sort_values("session").reset_index(drop=True)["target_position"].values
         for a in [3000, 10000, 30000]) / 3
stk = pd.read_csv(R/"submissions/stack_hl_session_ridge.csv"
                 ).sort_values("session").reset_index(drop=True)["target_position"].values

# 2. Blend
v2k3 = 0.80 * champ["target_position"].values + 0.20 * v2

# 3. Parkinson vol
bars = pd.concat([pd.read_parquet(R/"data/bars_seen_public_test.parquet"),
                  pd.read_parquet(R/"data/bars_seen_private_test.parquet")])
b0 = bars[bars.bar_ix < 50].sort_values(["session","bar_ix"]).copy()
b0["p_term"] = np.log(b0["high"] / b0["low"]) ** 2
park = np.sqrt(b0.groupby("session")["p_term"].mean() / (4 * np.log(2)))
park = park.reindex(champ["session"].values).fillna(park.median())
scaler = (park / park.median()) ** 1.2

# 4. Final formula
blend = 0.910 * (v2k3 - 0.28) + 0.12 * stk
y = blend / scaler.values
pd.DataFrame({"session": champ["session"].values,
              "target_position": y}).to_csv("submission.csv", index=False)
```
