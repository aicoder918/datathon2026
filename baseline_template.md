# Template + Direction Baseline

Simple, no-ML submission built from two per-session features:

1. **Template impact** — sum of train-estimated mean forward returns for each
   headline template that fires during bars 0–49 of the session.
2. **Direction 0→49** — `close[49] / close[0] - 1`, the signed return over the
   input window. This signal is *mean-reverting* (negatively correlated with
   the 50→99 target) on train data.

Position formula:

```
position = 1 + alpha * (z_template + beta * z_direction)
```

Train-side z-scores are applied at test time so public/private are on the
same scale as train.

## Inputs

- `hrt-eth-zurich-datathon-2026/data/bars_seen_train.parquet`
- `hrt-eth-zurich-datathon-2026/data/bars_unseen_train.parquet`
- `hrt-eth-zurich-datathon-2026/data/bars_seen_public_test.parquet`
- `hrt-eth-zurich-datathon-2026/data/bars_seen_private_test.parquet`
- `headline_features_train.parquet` (or `headline_features.parquet`)
- `headline_features_public.parquet`
- `headline_features_private.parquet`

## Generate

From the project root:

```bash
.venv/bin/python baseline_template.py --out submission_template_dir.csv
```

Default behavior: sweeps `(alpha, beta)` on in-sample train Sharpe and uses
the best combo for the submission (in our run: `alpha=0.5, beta=-1.0`).

To force specific values:

```bash
.venv/bin/python baseline_template.py \
  --out submission_template_dir.csv \
  --alpha 0.5 --beta -1.0
```

Conservative variant (half the bet size, same sign structure):

```bash
.venv/bin/python baseline_template.py \
  --out submission_template_dir_small.csv \
  --alpha 0.25 --beta -1.0
```

## Output format

```
session,target_position
<int>,<float>
...
```

20,000 rows = 10,000 public + 10,000 private sessions, sorted by session.

## Reference numbers (in-sample train)

| Setup                              | Sharpe |
|------------------------------------|--------|
| constant 1.0 (all long)            | 2.766  |
| template-only, alpha=0.25          | 2.863  |
| **template + direction, 0.5/-1.0** | **3.011** |

Signal correlations vs target:

| Feature                 | corr(feature, y) |
|-------------------------|------------------|
| template impact         | +0.039           |
| ret(0→49) direction     | −0.069           |
| template × direction    | +0.342           |

In-sample overstates public generalization. Public constant-long scored 2.17,
so in-sample 2.766 → public 2.17. By the same ratio, in-sample 3.011 would
project to ≈ 2.36 public, but direction’s mean-reversion edge may not
transfer as cleanly — submit and check.

## What the script does

1. `compute_template_impacts(bars_full, hls_train)` — per-template mean
   of `(close[t+5]/close[t]-1) - session_baseline` over all train headlines.
2. `session_template_score(hls, impacts)` — per-session sum across headlines.
3. `session_direction_0_49(bars)` — per-session `close[49]/close[0]-1`.
4. Train z-scoring stats (μ, σ) stored for both features.
5. Sweep `(alpha, beta)` over in-sample train Sharpe, pick best.
6. Apply same formula with train stats to public + private, write CSV.
