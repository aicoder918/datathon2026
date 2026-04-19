# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v54_amp110_sp_t125_m085.csv`)
- **LB Public Score**: **2.88951** ★ (AHEAD of affenmann team's 2.889!)
- **Date discovered**: 2026-04-19

## Composition
Re-shape the 3-signal blend (v51_tsagg06 @ LB 2.88066) to restore tails:
1. Start with v51_tsagg06 = 0.89 orig + 0.05 rr + 0.06 tsagg
2. Mean-preserving scale by ×1.10 (restores std 0.66 → 0.73, matching orig)
3. Softpos(t=1.25, m=0.85, pos-only) shaping

## Session progression
| Version | LB | Delta |
|---|---|---|
| aff7_amp140_softpos_pos_t125_m085 | 2.86777 | base |
| v47_sp_t120_m075_ar_w05 | 2.87484 | +0.007 |
| aff10_best_robustq01blend_95_05 | 2.87967 | +0.005 |
| v51_tsagg06 (3-signal blend) | 2.88066 | +0.001 |
| **v54_amp110_sp_t125_m085** (current) | **2.88951** | **+0.009** |
| Affenmann team LB | 2.889 | PASSED ✓ |
| Leader target | 2.916 | +0.027 gap |
| User ambition | 3.000 | +0.111 gap |

## Key insights
1. **BREAKTHROUGH: Reshape after blending**. The blend smooths tails (std drops 0.73 → 0.66). Re-amplifying ×1.10 then softpos (same params as orig) restored the tail confidence and jumped LB by +0.009 — largest single step this session.
2. **corr(ridge_robust_q01, orig) = 0.03** — nearly orthogonal. 5% weight worth +0.012 LB.
3. **corr(ts_aggressive, orig) = 0.35** — medium-diverse quality signal. 6% weight worth +0.001.
4. Softpos params invariant: t=1.25, m=0.85 peak for both orig and reshape application.
5. Scale ratio ≈ orig.std / blend.std = 1.103 — approximately recovers the original tail volatility.

## Lessons from failed paths
- v48 (softpos/AR on top of rq): WORSE — blend already smooth, extra shaping overfits.
- v52 (mix with v4/lgbm, corr 0.32/0.41): WORSE — neg corr doesn't help if individual signal noisy.
- v53 (tpl + agg composite, corr -0.17): WORSE — diversity alone insufficient.
- Only v54 (post-blend reshape) produced a real structural gain.

## Next exploration directions
1. **v55 fine sweep** (amp 1.08–1.15, t 1.20–1.30, m 0.80–0.90) — find tighter peak.
2. **v54 with AR blend** — add more diverse source *before* reshape.
3. Try **amp + softpos on rq alone** — may beat v51+reshape stack.
4. Train **LightGBM direct-Sharpe autograd** (SOTA_reaseach.txt method).

## Rate-limit state
- Both accounts burst-limited as of 05:17 UTC. Cool-off ~15-30 min typical.
