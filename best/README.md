# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v34_best_tstpl_w25.csv`)
- **LB Public Score**: **2.75293** ★
- **Date discovered**: 2026-04-19

## Composition
Blend:
- 75% × `v32_best_aff2_w45` (prior best, LB 2.72177)
- 25% × `ts_template_dir` (thinking-simple's template+direction signal)

## Top LB Scores
| Submission | LB |
|---|---|
| v34_best_tstpl_w25 | **2.75293** |
| v34_best_tstpl_w22 | 2.74981 |
| v33_best_tstpl_w20 | 2.74764 |
| v34_best_tstpl_w15 | 2.74186 |
| v32_best_aff2_w45 | 2.72177 |
| v35_best_ctx_w15 | 2.69242 |
| v27_meanens_top7 | 2.68924 |
| Leader target | 2.916 |

## Key insight
thinking-simple's `ts_template_dir` signal has a sweet spot around w=25%.
Going higher (w=30+) may overshoot. Need to test w=27, w=28, w=30.
aff4 diverse signals (ctx, ctxw, zsumw) underperformed — those don't complement.
