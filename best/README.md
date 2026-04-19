# Current Best LB Submission

## Current best
- **File**: `best.csv` (copy of `submissions/v30_best_aff2_rdg85_w20.csv`)
- **LB Public Score**: **2.71047** ★
- **Date discovered**: 2026-04-19

## Composition
Blend:
- 80% × `v27_meanens_top7` (our mean-ensemble of top 7 triple-blend ridges, LB 2.68924)
- 20% × `aff_tdvol_logstd_rdg85` (affenmann's new z-score template+direction+vol formula blended with their ridge at 85/15, LB 2.66383)

The key unlock was finding affenmann's NEW branch approach (their pipeline using
`pos = 1 + α*(z_template + β*z_direction + γ*z_vol)` with NEGATIVE β for mean-reversion).
Correlation with our best was only 0.73 → large ensemble gain.

## Top LB Scores
| Submission | LB |
|---|---|
| v30_best_aff2_rdg85_w20 | **2.71047** |
| v30_best_aff1_tdvol_w15 | 2.70620 |
| v30_best_aff2_rdg85_w15 | 2.70605 |
| v30_best_aff1_tdvol_w10 | 2.70127 |
| v30_best_affmean_w15 | 2.70110 |
| v27_meanens_top7 | 2.68924 |
| affenmann champion (old) | 2.667 |
| Leader target | 2.916 |
