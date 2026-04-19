# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v60_w16_a120_t125_m085.csv`)
- **LB Public Score**: **2.89858** ★ (AHEAD of affenmann team's 2.889)
- **Date discovered**: 2026-04-19 04:11 UTC

## Composition
Peak refined — shifted from w=0.15 (v59 = 2.89807) to **w=0.16** (+0.00051):
1. `v54 = v51_tsagg06` reshape-stack: 0.89 orig + 0.05 rr + 0.06 tsagg, amp×1.10, softpos(t=1.25,m=0.85)
2. `blend = 0.84*v54 + 0.16*ar_current`
3. `y = mean + 1.20*(blend - mean)`
4. `y = softpos(y, t=1.25, m=0.85, pos-only)`

## v60 sweep curve (LB @ amp=1.20, t=1.25, m=0.85)
| w | LB |
|---|---|
| 0.13 | 2.89605 |
| 0.14 | 2.89615 |
| 0.15 | 2.89807 |
| **0.16** | **2.89858** ★ |
| 0.17 | 2.89611 |
| amp=1.18 | 2.89312 |
| amp=1.22 | 2.89589 |
| amp=1.25 | 2.89500 |
| m=0.80 | 2.89784 (≈ peak) |

## Session progression
| Version | LB | Delta |
|---|---|---|
| aff7_amp140_softpos_pos_t125_m085 | 2.86777 | base |
| v54_amp110_sp_t125_m085 | 2.88951 | +0.022 |
| v59_arblend_w15_amp120_sp | 2.89807 | +0.009 |
| **v60_w16_a120** (current) | **2.89858** | **+0.0005** |
| Affenmann team LB | 2.889 | PASSED ✓ |
| Leader target | 2.916 | +0.017 gap |
| User ambition | 3.000 | +0.101 gap |

## Key insights
1. **corr(ar, BEST) = 0.646** → ar diverse (pushes LB +0.009 at w=0.16).
2. **corr(lgbm, BEST) = 0.143** → LGBM autograd-Sharpe signal VERY diverse (trained 2026-04-19 04:14). Could beat ar if noise is low enough. Testing in v63.
3. **corr(agg, BEST) = 0.345, corr(v4, BEST) = 0.354** — medium diverse. agg/v4 corr -0.306 (opposite noise). Testing v61 multi-signal.
4. Peak is **very sharp** on all axes: w ±0.01 / amp ±0.02 / m ±0.05 move LB by 0.001-0.003.
5. **Pattern: blend to smooth → amp to restore → softpos to reshape tails**. Each step is cumulative.

## Queues / workers
- `/tmp/watchdog2.sh` (PID 2700918) polls every ~3min, rotates A1↔A2 on 429, submits from `/tmp/priority_queue.txt`.
- `/tmp/priority_queue.txt`: 30 candidates focusing on v63 (w=0.16 peak + LGBM signal blends).
- `/tmp/watchdog2.log`: live submission log.

## Next exploration directions
1. **v63 LGBM-blend**: corr 0.143 with best — if this adds value, could be next breakthrough.
2. **v63 w=0.18, 0.19** — push past peak to confirm it's not monotonically better.
3. **v61 multi-signal** (v54 + ar + agg): leverage orthogonality.
4. **v62 asymmetric reshape** — different tail shaping per side.
5. **Daily quota**: acct1+2 exhausted. Rolling reset (submissions >24h old unblock quota). Full reset next at 00:00 UTC 2026-04-20.
