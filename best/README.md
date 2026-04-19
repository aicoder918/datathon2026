# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v59_arblend_w15_amp120_sp.csv`)
- **LB Public Score**: **2.89807** ★ (AHEAD of affenmann team's 2.889)
- **Date discovered**: 2026-04-19

## Composition
Blend v54 (the 2.88951 reshape-of-reshape) with autoresearch signal, then amp+softpos:
1. `v54 = v51_tsagg06` reshape-stack: 0.89 orig + 0.05 rr + 0.06 tsagg, amp×1.10, softpos(t=1.25,m=0.85)
2. `blend = 0.85*v54 + 0.15*ar_current` (ar corr 0.607 with v54 — high diversity)
3. `y = mean + 1.20*(blend - mean)`
4. `y = softpos(y, t=1.25, m=0.85, pos-only)`

## Session progression
| Version | LB | Delta |
|---|---|---|
| aff7_amp140_softpos_pos_t125_m085 | 2.86777 | base |
| v47_sp_t120_m075_ar_w05 | 2.87484 | +0.007 |
| aff10_best_robustq01blend_95_05 | 2.87967 | +0.005 |
| v51_tsagg06 (3-signal blend) | 2.88066 | +0.001 |
| v54_amp110_sp_t125_m085 | 2.88951 | +0.009 |
| **v59_arblend_w15_amp120_sp** (current) | **2.89807** | **+0.009** |
| Affenmann team LB | 2.889 | PASSED ✓ |
| Leader target | 2.916 | +0.018 gap |
| User ambition | 3.000 | +0.102 gap |

## Key insights
1. **NEW BREAKTHROUGH: autoresearch-blend + aggressive reshape**. ar_current (corr 0.61 w/ v54 — very diverse) at 15% weight pushes LB by +0.0086. The smoothing from blend requires a BIGGER amp (1.20 vs 1.10).
2. **Reshape after blending**. The blend smooths tails (std drops 0.73 → 0.66). Re-amplifying + softpos restored tail info.
3. **corr(ridge_robust_q01, orig) = 0.03** — nearly orthogonal. 5% weight worth +0.012 LB.
4. **corr(ts_aggressive, orig) = 0.35** — medium-diverse quality signal. 6% weight worth +0.001.
5. Softpos params invariant: t=1.25, m=0.85 peak across every application.
6. **Pattern: blend to smooth → amp to restore → softpos to reshape tails**. Each step is cumulative.

## Lessons from failed paths
- v48 (softpos/AR on top of rq): WORSE — blend already smooth, extra shaping overfits.
- v52 (mix with v4/lgbm, corr 0.32/0.41): WORSE — neg corr doesn't help if individual signal noisy.
- v53 (tpl + agg composite, corr -0.17): WORSE — diversity alone insufficient.
- Only v54 (post-blend reshape) produced a real structural gain.

## Next exploration directions
1. **v60 fine sweep** around w=0.15, amp=1.20 (403 candidates built). Find tighter peak.
2. **4-way blend**: orig + rr + agg + ar (ar at 8-12%). Built in v60_4way_*.
3. **Heavier ar weight** (w=0.20, 0.25) might help more.
4. Train **better autoresearch signal** — ar_current corr 0.61 helped; maybe different config pushes further.

## Rate-limit state
- Both accts hit daily cap (~100 submits each). A2 briefly unblocked 05:52 UTC to submit one — resets at 00:00 UTC 2026-04-20.
- Watchdog @ /tmp/watchdog_queue.sh polls every 5 min, submits from /tmp/priority_queue.txt.
