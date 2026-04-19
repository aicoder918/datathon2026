# Priority Submissions (for when Kaggle rate-limit clears)

Current best: **v33_best_tstpl_w20 → LB 2.74764**

## Top priority (most diverse → biggest expected LB jump)

| # | File | corr_best | Rationale |
|---|---|---|---|
| 1 | `v35_best_ctx_w15` | 0.89 | aff4 template_ctx (corr 0.14 alone) at mod weight |
| 2 | `v35_best_ctx_w20` | 0.86 | push ctx weight to 20% |
| 3 | `v35_best_ctx_w25` | 0.80 | push ctx weight to 25% — most diverse binary |
| 4 | `v35_best_ctxw_w20` | 0.89 | ctx_winner variant |
| 5 | `v35_best_zsumw_w20` | 0.88 | zsum_dir49 (corr 0.34 alone) |
| 6 | `v35_q70_ctx15_zsw10_tid5` | 0.82 | quad with 3 diverse aff4 sources |
| 7 | `v35_b65_ctx20_zsumw15` | 0.76 | maximally diverse triple — high risk/reward |
| 8 | `v34_best_tstpl_w25` | 0.9996 | push thinking-simple weight slightly |
| 9 | `v34_best_tstpl_w15` | 0.9996 | pull thinking-simple weight back |
| 10 | `v38_best_meanens_w30` | 0.95 | mean-of-5 diverse v35 candidates |

## Submission strategy
1. Submit #1-#5 first (each exploits different diverse signal).
2. Based on which beats 2.74764, extend to #6-#10 focusing on the winning signal.
3. Build v39 with higher weight on the winning signal type.

## Rate-limit notes
- Kaggle daily cap is ~50 submissions. Used 45/50 today as of 2026-04-19 01:08 UTC.
- Daily reset at UTC midnight → ~23h wait.
- Burst throttle may clear sooner.
