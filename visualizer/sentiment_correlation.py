"""Does FinBERT sentiment on a headline predict the subsequent price move?"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

bars = pd.concat([
    pd.read_parquet(DATA_DIR / "bars_seen_train.parquet"),
    pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet"),
], ignore_index=True).sort_values(["session", "bar_ix"]).reset_index(drop=True)

hl = pd.concat([
    pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet"),
    pd.read_parquet(DATA_DIR / "headlines_unseen_train.parquet"),
], ignore_index=True)

sent = pd.read_parquet(DATA_DIR / "headlines_finbert_sentiment.parquet")
hl = hl.merge(sent, on="headline", how="left")
hl["sent_int"] = hl["label"].map({"positive": 1, "negative": -1, "neutral": 0})

# Build a (session, bar_ix) -> close lookup
close = bars.set_index(["session", "bar_ix"])["close"].sort_index()
max_bar_per_session = bars.groupby("session")["bar_ix"].max()


def forward_return(row, k):
    sid, b = row["session"], row["bar_ix"]
    end = min(b + k, max_bar_per_session[sid])
    try:
        c0 = close.loc[(sid, b)]
        c1 = close.loc[(sid, end)]
    except KeyError:
        return np.nan
    return c1 / c0 - 1


HORIZONS = [1, 3, 5, 10]
for k in HORIZONS:
    hl[f"ret_{k}"] = hl.apply(lambda r: forward_return(r, k), axis=1)

# ---------- Plot 1: forward-return distributions per sentiment ----------
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4 * len(HORIZONS), 4), sharey=True)
colors = {"positive": "#2ca02c", "neutral": "#888888", "negative": "#d62728"}
for ax, k in zip(axes, HORIZONS):
    data = [hl.loc[hl["label"] == lab, f"ret_{k}"].dropna() for lab in ["negative", "neutral", "positive"]]
    parts = ax.boxplot(data, labels=["negative", "neutral", "positive"], showfliers=False, patch_artist=True)
    for patch, lab in zip(parts["boxes"], ["negative", "neutral", "positive"]):
        patch.set_facecolor(colors[lab])
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Forward return over {k} bar(s)")
    ax.set_xlabel("FinBERT label")
axes[0].set_ylabel("close[t+k] / close[t] - 1")
fig.suptitle("Headline sentiment → subsequent price return", y=1.02)
fig.tight_layout()
fig.savefig(str(OUT_DIR / "sentiment_forward_returns.png"), dpi=130, bbox_inches="tight")

# ---------- Plot 2: session-level — net sentiment vs total session return ----------
session_return = bars.groupby("session")["close"].agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
session_sent = hl.groupby("session")["sent_int"].sum().reindex(session_return.index, fill_value=0)
session_n = hl.groupby("session").size().reindex(session_return.index, fill_value=0)
session_avg_sent = (session_sent / session_n.replace(0, np.nan)).fillna(0)

corr_sum = np.corrcoef(session_sent, session_return)[0, 1]
corr_avg = np.corrcoef(session_avg_sent, session_return)[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(session_sent, session_return, s=8, alpha=0.4)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axvline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("net sentiment (sum of +1/-1 per headline)")
axes[0].set_ylabel("session return (last/first - 1)")
axes[0].set_title(f"Session: net sentiment vs return  (corr={corr_sum:.3f})")

axes[1].scatter(session_avg_sent, session_return, s=8, alpha=0.4)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].axvline(0, color="black", linewidth=0.5)
axes[1].set_xlabel("average sentiment per headline")
axes[1].set_ylabel("session return")
axes[1].set_title(f"Session: avg sentiment vs return  (corr={corr_avg:.3f})")
fig.tight_layout()
fig.savefig(str(OUT_DIR / "sentiment_session_corr.png"), dpi=130)

# ---------- Plot 4: forward return as a function of bars elapsed ----------
MAX_K = 50
ks = list(range(1, MAX_K + 1))

# vectorized forward-return computation across many horizons
hl_idx = hl[["session", "bar_ix"]].copy()
hl_idx["c0"] = close.reindex(list(zip(hl_idx["session"], hl_idx["bar_ix"]))).values
curves = {lab: [] for lab in ["negative", "neutral", "positive"]}
errs = {lab: [] for lab in ["negative", "neutral", "positive"]}
for k in ks:
    end_bars = np.minimum(hl_idx["bar_ix"] + k,
                          hl_idx["session"].map(max_bar_per_session))
    c1 = close.reindex(list(zip(hl_idx["session"], end_bars))).values
    ret = c1 / hl_idx["c0"].values - 1
    for lab in curves:
        mask = (hl["label"] == lab).values & np.isfinite(ret)
        r = ret[mask]
        curves[lab].append(r.mean())
        errs[lab].append(1.96 * r.std() / np.sqrt(len(r)))

fig, ax = plt.subplots(figsize=(10, 5))
for lab in ["negative", "neutral", "positive"]:
    m = np.array(curves[lab])
    e = np.array(errs[lab])
    ax.plot(ks, m, color=colors[lab], label=lab, linewidth=1.8)
    ax.fill_between(ks, m - e, m + e, color=colors[lab], alpha=0.15)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel("bars after headline (k)")
ax.set_ylabel("mean forward return: close[t+k]/close[t] - 1")
ax.set_title("Forward-return drift after a headline, by FinBERT label  (band = 95% CI)")
ax.legend()
fig.tight_layout()
fig.savefig(str(OUT_DIR / "sentiment_return_vs_bars.png"), dpi=130)


# ---------- Print a numeric summary ----------
print("Forward-return mean by sentiment label:")
print(hl.groupby("label")[[f"ret_{k}" for k in HORIZONS]].mean().round(5))
print()
print(f"Session-level Pearson correlation (net sentiment vs session return):  {corr_sum:.4f}")
print(f"Session-level Pearson correlation (avg sentiment vs session return):  {corr_avg:.4f}")
print(f"Saved 3 figures to {OUT_DIR}")
