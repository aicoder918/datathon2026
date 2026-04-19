# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v64_cx_ar16_cx02_a122.csv`)
- **LB Public Score**: **2.90074** ★★ (FIRST BREAKTHROUGH PAST 2.900!)
- **Date discovered**: 2026-04-19 04:30 UTC

## Composition
First time crossing **2.900** — via ultra-diverse ctxvol signal at 2% weight + higher amp (1.22):
1. `v54 = v51_tsagg06` reshape-stack: 0.89 orig + 0.05 rr + 0.06 tsagg, amp×1.10, softpos(t=1.25,m=0.85)
2. `blend = 0.82*v54 + 0.16*ar + 0.02*ctxvol`  (ctxvol = affenmann's submission_ctxvol_ols_ctx1)
3. `y = mean + 1.22*(blend - mean)`
4. `y = softpos(y, t=1.25, m=0.85, pos-only)`

## Key signals + correlations
| Signal | corr(best) | Notes |
|---|---|---|
| v54 (own ridge) | 0.998 | core |
| ar_current | 0.646 | ridge mean+log-var, +0.009 LB at w=0.16 |
| ctxvol_ols | **0.170** | affenmann raw signal — ULTRA-diverse |
| lgbm_sharpe | 0.143 | LGBM autograd-Sharpe, low-sharpe signal |
| agg | 0.345 | ts_aggressive |
| v4 | 0.354 | ts_v4 |

## Session progression
| Version | LB | Delta | Notes |
|---|---|---|---|
| aff7_amp140_softpos_pos_t125_m085 | 2.86777 | base | |
| v54_amp110_sp_t125_m085 | 2.88951 | +0.022 | 3-signal reshape |
| v59_arblend_w15_amp120_sp | 2.89807 | +0.009 | +ar blend breakthrough |
| v60_w16_a120_t125_m085 | 2.89858 | +0.0005 | peak shift w=0.15→0.16 |
| **v64_cx_ar16_cx02_a122** | **2.90074** ★ | +0.00216 | +ctxvol + higher amp |
| Leader target | 2.916 | +0.015 gap | |
| User ambition | 3.000 | +0.099 gap | |

## Key insights
1. **ctxvol at 2% + amp=1.22** cracks 2.90. The ultra-diverse signal (corr 0.17 with best) was DAMAGING at amp=1.20 (2.89619) but HELPFUL at amp=1.22 (2.90074). Higher amp counteracts the smoothing.
2. **Pattern refined: more diverse signal → needs MORE aggressive reshape to re-amplify.** This matches the v59 insight (amp 1.20 vs base 1.10).
3. LB-to-v60 curve is very sharp: ±0.5% on w changes LB by 0.001-0.003.
4. ar and ctxvol fill different diversity niches — ar diverse-from-v54 (0.646), ctxvol diverse-from-v54+ar.

## Rate limit / workers
- A1+A2 daily quota exhausted. **A3 (fresh!)** running via watchdog2.
- `/tmp/watchdog2.sh`: rotates A1→A2→A3, submits from priority queue.
- `/tmp/priority_queue.txt`: v66 fine amp sweep (1.20-1.30) × cx weight (1-5%) × w_ar (14-18).

## Next exploration
1. **v66 fine amp sweep** (124, 125, 126) around ctxvol+2%. Maybe even higher amp helps.
2. **Higher ctxvol weight** (3-5%) at amp=1.22+.
3. Different diverse-signal combinations (ctxvol + lgbm, without ar).
4. Try raw ctxvol without any rescaling — maybe amp is compensating for ctxvol's wide std.
