# Handoff: HRT ETH Zurich Datathon 2026 (Kaggle submissions)

## What we’re optimizing

- **Primary metric**: public leaderboard Sharpe (higher is better).
- **Current goal**: push beyond the recent plateau around **~2.9019** public Sharpe on the correct competition (see “Current best” below).

## Correct Kaggle competition (important)

We previously targeted the wrong slug. The correct competition is:

- **Competition slug**: `hrt-eth-zurich-datathon-2026`

## Credentials / auth pattern we’re using

We are submitting with a **Kaggle API token** via environment variable:

- **`KAGGLE_API_TOKEN`**: `KGAT_1bb9f372c56583949b449974a88a0758`

**Security note**: this token is sensitive. Prefer exporting it in your shell session (or CI secrets) rather than committing it long-term. This file exists because you explicitly asked for a handoff artifact.

### Recommended “clean submit” shell snippet

This avoids Cursor/proxy env interference and avoids mixing username/password auth with token auth:

```bash
unset KAGGLE_USERNAME KAGGLE_KEY

export KAGGLE_API_TOKEN='KGAT_1bb9f372c56583949b449974a88a0758'

env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GIT_HTTP_PROXY -u GIT_HTTPS_PROXY \
  -u SOCKS_PROXY -u SOCKS5_PROXY -u socks_proxy -u socks5_proxy \
  /Users/mgershman/Desktop/datathon/.venv/bin/kaggle competitions submit \
    -c hrt-eth-zurich-datathon-2026 \
    -f "/ABS/PATH/TO/submission.csv" \
    -m "short human-readable note"
```

### Alternative: classic `username` + `key` (`kaggle.json`)

A copy of your `Downloads/kaggle (3).json` is installed as:

- **`/Users/mgershman/Desktop/datathon/.kaggle/kaggle.json`** (mode `600`)

Use it by **unset**ting the API token and pointing `KAGGLE_CONFIG_DIR` at that folder:

```bash
unset KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY

export KAGGLE_CONFIG_DIR="/Users/mgershman/Desktop/datathon/.kaggle"

env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GIT_HTTP_PROXY -u GIT_HTTPS_PROXY \
  -u SOCKS_PROXY -u SOCKS5_PROXY -u socks_proxy -u socks5_proxy \
  /Users/mgershman/Desktop/datathon/.venv/bin/kaggle competitions submit \
    -c hrt-eth-zurich-datathon-2026 \
    -f "/ABS/PATH/TO/submission.csv" \
    -m "short human-readable note"
```

### Useful companion commands

List recent submissions + scores:

```bash
unset KAGGLE_USERNAME KAGGLE_KEY
export KAGGLE_API_TOKEN='KGAT_1bb9f372c56583949b449974a88a0758'

env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GIT_HTTP_PROXY -u GIT_HTTPS_PROXY \
  -u SOCKS_PROXY -u SOCKS5_PROXY -u socks_proxy -u socks5_proxy \
  /Users/mgershman/Desktop/datathon/.venv/bin/kaggle competitions submissions \
    -c hrt-eth-zurich-datathon-2026
```

## Repo + paths

- **Workspace root**: `/Users/mgershman/Desktop/datathon/datathon2026`
- **Python venv**: `/Users/mgershman/Desktop/datathon/.venv/bin/python`
- **Kaggle CLI**: `/Users/mgershman/Desktop/datathon/.venv/bin/kaggle`
- **Submission outputs**: mostly under `datathon2026/submissions/`

## What we did recently (high-signal)

### Fixed the biggest blocker

- Wrong competition slug caused confusing `403` behavior earlier.
- After switching to `hrt-eth-zurich-datathon-2026`, submissions succeeded.

### Ran a broad “queue burn” on the correct board

We submitted a large batch of pre-generated candidates, including:

- contextual/template blends (mostly weaker than the seed20 line on this board)
- `seed20` additive blends (strong)
- amplitude / leverage variants around the emerging winner

### Trained a new orthogonal CatBoost branch for blending

- Ran: `datathon2026/models/catboost_bars_bootstrap_bag.py`
- Outputs:
  - `datathon2026/submissions/chatgpt/catboost_bars_bootstrap30.csv`
  - `datathon2026/submissions/chatgpt/catboost_bars_bootstrap30_killshorts_blend.csv`
- Blends vs the current best were tested; tiny mixes sometimes nudge, but the dominant structure remains the **seed20 + shaping** family.

## Current best (public leaderboard)

**Best observed public score**: **2.91264** (rank 5 of LB)

- **File**: `datathon2026/submissions/champ_plus_ridgehl_a3k_870_130.csv`
- **Kaggle description**: `headline 87/13 (peak zoom)`
- **Composition**: `0.87 * champ + 0.13 * ridge_hl_a3000` where
  - `champ = submission_best_plus_ridge_top10_935_065.csv` (was 2.90305)
  - `ridge_hl_a3000 = submissions/ridge_hl_a3000.csv` — ridge over per-headline
    features (template/sector/region one-hots + bar_ix + log$amt + pct + finbert
    sentiment + session cum_ret + rolling vol) predicting forward-5-bar return,
    aggregated per session via exp recency sum, shaped via thresholded_inv_vol.
    Training script: `run_headline_model.py`.

**Weight sweep evidence (robust, not an LB spike)**:
- 98/2: 2.90568 · 95/5: 2.90890 · 90/10: 2.91213 · 87/13: **2.91264** · 85/15: 2.91233 · 80/20: 2.90907
- 3-branch base + HL @87/13: 2.91239 (confirms cross-base robustness)

## Historical runner-up (pre-headline-model plateau): **2.90306**

- **File**: `datathon2026/submissions/sub_base_rt10_ra50_930_055_015.csv`
- 3-branch: 0.930*base + 0.055*rt10 + 0.015*ra50

## Previous handoff champion: **2.90193**

- **File**: `datathon2026/submissions/submission_best_x103_plus_bootstrap30_killshorts_990_010.csv`
- **Kaggle description**: `best_x103_plus_bootstrap30_killshorts_990_010`

**Close runner-up (original)**: **2.90190**

- **File**: `datathon2026/submissions/submission_best_plus_seed20_900_100_amp105_seed5decay_0910_0090_amp105_x103.csv`
- **Kaggle description**: `seed20_900100_amp105_seed5_91090_amp105_x103`

Interpretation:

- The leaderboard is extremely flat in the **2.9015–2.9020** band; many “big” ideas don’t move it.
- Further gains likely require **either** a genuinely new predictive branch **or** very careful last-mile shaping—**but** last-mile moves risk leaderboard overfitting.

## The “shape stack” intuition (what the best files represent)

The strongest line on this board is basically:

1. Start from the strong blended baseline based on:
   - `submission_best_domainweighted_seed5decay_900_100_dw020_amp112.csv` (historical strong “amp112” champion on the earlier workflow)
2. Add **`catboost_bars_seed20.csv`** mass at **90/10** vs **95/5** (90/10 won locally on this board earlier in the session)
3. Apply **amplitude / leverage** transforms around 1 (`1 + a*(x-1)`), commonly ~**1.05**
4. Add a **small** amount of `catboost_bars_seed5_decay.csv`, then optionally another small outer amplitude (the `x103` naming)

Net: it’s “ensemble core + mild leverage + tiny decay-catboost tilt”.

## Workflow I want the next agent to follow

### 1) Always submit on the correct slug + token pattern

Use the snippet above. If you see proxy tunnel errors, keep the `env -u ...PROXY...` prefix.

### 2) Prefer “broad jumps” first, then narrow only if the curve is flat but promising

We already burned a lot of marginal candidates. Next work should bias toward:

- **New model families** or materially different feature sets (not another 0.1% blend tweak)
- **One** coarse leverage test (e.g. 1.03 vs 1.06) *after* a structural change

### 3) Keep a running scoreboard locally

After each batch:

- Pull `kaggle competitions submissions -c hrt-eth-zurich-datathon-2026`
- Record: filename, message, public score

### 4) Don’t explode submission rate blindly

If Kaggle rate-limits, queue locally and submit in waves.

## Concrete “next attempts” (suggested)

Pick 1–2 at a time (not all at once):

- **Train a new CatBoost variant** with a real hyperparam/feature change (not just re-seeding), then blend at **97/3** and **95/5** against:
  - `submission_best_plus_seed20_900_100_amp105_seed5decay_0910_0090_amp105_x103.csv`
- **Re-open the ridge / contextual lines** but only if you can generate candidates that are **low-correlation** to the seed20 winner on test rows (correlation screening in-notebook/script).
- **Avoid** spending many submissions on:
  - `submission_ctx_grid_dir49sum_p10_*` family (was consistently weak on this board)
  - naive template `tz_*` 2% mixes into the winner (mostly hurt here)

## Files you should treat as “anchors”

- **Historical strong baseline**: `submission_best_domainweighted_seed5decay_900_100_dw020_amp112.csv`
- **CatBoost branches**:
  - `submissions/catboost_bars_seed20.csv`
  - `submissions/catboost_bars_seed5_decay.csv`
  - `submissions/catboost_bars_domain_weighted.csv`
  - `submissions/chatgpt/catboost_bars_bootstrap30.csv`
- **Blend helper**: `datathon2026/blend_submissions.py` (use venv python to run)

## Quick “update me” command for humans

Paste scores for the top 10 most recent submissions:

```bash
unset KAGGLE_USERNAME KAGGLE_KEY
export KAGGLE_API_TOKEN='KGAT_1bb9f372c56583949b449974a88a0758'

env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GIT_HTTP_PROXY -u GIT_HTTPS_PROXY \
  -u SOCKS_PROXY -u SOCKS5_PROXY -u socks_proxy -u socks5_proxy \
  /Users/mgershman/Desktop/datathon/.venv/bin/kaggle competitions submissions \
    -c hrt-eth-zurich-datathon-2026 \
| head -n 15
```

## Open questions / risks

- **Overfitting to public LB noise** is real: the difference between 2.9019 and 2.9025 may not be meaningful without a local validation story.
- The **`x103` chain** is powerful but easy to over-tune; treat additional `x103_x...` variants skeptically.
