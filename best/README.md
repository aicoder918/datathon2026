# Current Best LB Submission

Updated automatically on each new LB best discovery.

## Current best
- **File**: `best.csv` (copy of `submissions/v27_meanens_top7.csv`)
- **LB Public Score**: **2.68924**
- **Date discovered**: 2026-04-19

## Composition
Mean ensemble (equal-weight averaging) of top 7 triple-blend candidates:
1. `v24_triple_32_38_30.csv` (LB 2.68831)
2. `v23_triple_v18_30_tq55_50_a10k_20.csv` (LB 2.68820)
3. `v23_triple_v18_30_tq55_40_a10k_30.csv` (LB 2.68805)
4. `v24_triple_28_42_30.csv` (LB 2.68758)
5. `v23_triple_v18_40_tq55_30_a10k_30.csv` (LB 2.68708)
6. `v23_blend_v18_v21tq55_w40.csv` (LB 2.68688)
7. `v23_blend_v18_v21tq55_w45.csv` (LB 2.68660)

Each component is itself a blend of v18 (my pipeline) + v21_a20k_tq55 + v21_a10k (affenmann ridge variants).
The super-ensemble cancels noise across near-optimal blend weight choices, giving +0.001 over single-best.

## History (top LB scores)
| Submission | LB |
|---|---|
| v27_meanens_top7 | **2.68924** |
| v27_t_36_38_26 | 2.68885 |
| v27_t_34_38_27 | 2.68866 |
| v27_t_32_40_27 | 2.68855 |
| v27_meanens_top3 | 2.68844 |
| v27_t_34_36_29 | 2.68836 |
| v24_triple_32_38_30 | 2.68831 |
| affenmann champion | 2.667 |
| v18 baseline | 2.517 |
| Leader (target) | 2.916 |
