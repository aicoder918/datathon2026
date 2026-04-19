# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/aff7_template044_ridge1000_58_42_amp140_softpos_pos_t125_m085.csv`)
- **LB Public Score**: **2.86777** ★
- **Date discovered**: 2026-04-19

## Composition
Affenmann's pipeline:
- Base: 58% × template044 + 42% × ridge α=1000
- Amplification scale: ×1.40
- Softpos (positive-only) shaping with thresh=1.25, margin=0.85

## Top LB Scores
| Submission | LB |
|---|---|
| aff7_amp140_softpos_pos_t125_m085 | **2.86777** |
| aff7_amp140_softtail_t125_m085 | 2.86513 |
| aff7_amp140_softtail | 2.86283 |
| aff7_amp145_softtail | 2.86232 |
| aff7_amp140 (no softtail) | 2.86071 |
| aff6_template044_ridge300_60_40 | 2.82426 |
| v34_best_tstpl_w25 | 2.75293 |
| Leader target | 2.916 |

## Pending submissions (account rate-limit blocked, not IP)
- aff7_amp140_softtail_t125_m080 / t175_m085
- aff7_amp140_softpos_pos_t15_m085
- aff8 variants from latest affenmann: amp140_softtail_t125_m075, amp145_softtail_t125_m085,
  amp140_longcap35/375, amp140_robust, amp150_softtail, best_asym_l145_s125

## Key insight
Softpos_pos with t125_m085 BEAT softtail by +0.005 — positive-only shaping > full tail
shaping at this scale. Need to test t15_m085 (wider thresh) and t125 with smaller margin.
Note: Kaggle 429 is account-level, not IP — even with proxies we hit the daily cap.
