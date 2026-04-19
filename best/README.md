# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v33_best_tstpl_w20.csv`)
- **LB Public Score**: **2.74764** ★
- **Date discovered**: 2026-04-19

## Composition
Blend:
- 80% × `v32_best_aff2_w45` (our prior best, LB 2.72177)
- 20% × `ts_template_dir` (thinking-simple's template+direction signal, corr 0.81)

## Top LB Scores
| Submission | LB |
|---|---|
| v33_best_tstpl_w20 | **2.74764** |
| v32_best_aff2_w45 | 2.72177 |
| v32_best_aff2_w50 | 2.72159 |
| v31_best_aff2_rdg85_w40 | 2.72108 |
| v32_b55_a1_10_a2_35 | 2.72097 |
| v32_best_aff2_w55 | 2.72050 |
| v33_best_tsproxy_w20 | 2.71352 |
| v33_best_tsproxy_w30 | 2.70692 |
| v33_best_tslgbm_w15 | 2.69944 |
| v27_meanens_top7 | 2.68924 |
| ts_proxy_fit (alone) | 2.60972 |
| Leader target | 2.916 |

## Key insight
The thinking-simple `ts_template_dir` signal at moderate weight (w=0.20) is
a massively diverse ingredient: weak alone but complementary to our ridge+aff2 blend.
Corr 0.81 with current best, yet adds +0.026 LB when blended at 20%.
