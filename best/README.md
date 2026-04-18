# Current Best LB Submission

Updated automatically on each new LB best discovery.

## Current best
- **File**: `best.csv` (copy of `submissions/v24_triple_32_38_30.csv`)
- **LB Public Score**: **2.68831**
- **Date discovered**: 2026-04-19

## Composition
Triple blend at session level:
- 32% × `v18` (my own pipeline, LB 2.517 baseline)
- 38% × `v21_a20k_tq55_sa10_fl030` (affenmann ridge α=20000, threshold q=0.55, sa=1.0, floor=0.30)
- 30% × `v21_a10k_tq50_sa10_fl030` (affenmann ridge α=10000, threshold q=0.50, sa=1.0, floor=0.30)

All three components share the same finalize pipeline (`scaled = pred/mean(|pred|)`, blend with constant-long via `shrink_alpha`, floor shorts).

## History (top LB scores so far)
| Submission | LB |
|---|---|
| v24_triple_32_38_30 | **2.68831** |
| v23_triple_v18_30_tq55_50_a10k_20 | 2.68820 |
| v23_triple_v18_30_tq55_40_a10k_30 | 2.68805 |
| v24_triple_28_42_30 | 2.68758 |
| v23_triple_v18_40_tq55_30_a10k_30 | 2.68708 |
| v23_blend_v18_v21tq55_w40 | 2.68688 |
| affenmann champion (LB 2.667) | 2.667 |
| v18 baseline | 2.517 |
| Leader (target) | 2.916 |
