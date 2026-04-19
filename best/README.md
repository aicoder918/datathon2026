# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v51_tsagg06.csv`)
- **LB Public Score**: **2.88066** ★
- **Date discovered**: 2026-04-19

## Composition
3-signal weighted blend (orig_best + ridge_robust + thinking-simple-aggressive):
- 0.89 × orig_best (= aff7_amp140_softpos_pos_t125_m085, LB 2.86777 alone)
- 0.05 × ridge_robust_q01 (winsorized-1/99 RobustScaler RidgeCV; corr 0.03 with orig — nearly orthogonal)
- 0.06 × thinking-simple-aggressive (lgbm/stacked pipeline from thinking-simple branch; corr 0.35 with orig)

## Session progression
| Version | LB | Delta |
|---|---|---|
| aff7_amp140_softpos_pos_t125_m085 | 2.86777 | base |
| v47_sp_t120_m075_ar_w05 | 2.87484 | +0.007 |
| aff10_best_robustq01blend_95_05 | 2.87967 | +0.005 |
| **v51_tsagg06** (current) | **2.88066** | +0.001 |
| Affenmann team LB | 2.889 | +0.008 gap |
| Leader target | 2.916 | +0.035 gap |
| User ambition | 3.000 | +0.119 gap |

## Key insights
1. **corr(ridge_robust_q01, orig_best) = 0.03** — nearly orthogonal. 5% weight drove +0.012 LB (biggest single step).
2. **corr(thinking-simple-aggressive, orig_best) = 0.35, std=0.26** — decent quality + diverse. 6% weight drove +0.001.
3. **Softpos formula** (reverse-engineered): `pos > 1+t → m*pos + (1-m)`. Peak t=1.20–1.25, m=0.75–0.85 — ALREADY baked into orig.
4. **Failed bets**:
   - v48: softpos/AR on top of rq → WORSE (blend already smooths tails; re-shaping overfits)
   - v52: blending v4/lgbm signals at corr 0.32/0.41 → WORSE (signals individually too noisy despite diversity)
   - AR (autoresearch) signals at corr 0.59–0.61 → modest gains (+0.003) before plateau
5. **Kaggle 429 is account-level**. Backup token unlocks burst submissions.

## Rate-limit state
- Account 1 (KGAT_4671170): active, currently used after Account 2 hit burst cap
- Account 2 (KGAT_934fa6bd): hit burst cap at ~17 submissions; cools in minutes

## Next exploration directions (to break past 2.881 plateau)
1. Retrain autoresearch with **Adjusted Sharpe Ratio** (penalize skew/kurtosis) — see SOTA_reaseach.txt
2. Fresh NLP embeddings via sentence-transformer + PCA (current uses FinBERT-only)
3. Direct Sharpe optimization in LightGBM via PyTorch autograd (SOTA method)
4. Monitor affenmann for next score push (they're ahead at 2.889)
