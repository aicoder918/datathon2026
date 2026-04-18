"""Distribution of how many headlines each session has."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

hl_seen = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
hl_unseen = pd.read_parquet(DATA_DIR / "headlines_unseen_train.parquet")

seen_counts = hl_seen.groupby("session").size()
unseen_counts = hl_unseen.groupby("session").size()
total_counts = seen_counts.add(unseen_counts, fill_value=0).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, data, title, color in [
    (axes[0], seen_counts, "Seen half (bars 0–49)", "#1f77b4"),
    (axes[1], unseen_counts, "Unseen half (bars 50–99)", "#d62728"),
    (axes[2], total_counts, "Full session", "#444444"),
]:
    ax.hist(data, bins=range(int(data.min()), int(data.max()) + 2),
            color=color, alpha=0.8, edgecolor="white")
    ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean={data.mean():.1f}")
    ax.axvline(data.median(), color="black", linestyle=":", linewidth=1,
               label=f"median={data.median():.0f}")
    ax.set_title(f"{title}  (n={len(data)} sessions)")
    ax.set_xlabel("headlines per session")
    ax.legend(fontsize=8)
axes[0].set_ylabel("number of sessions")
fig.suptitle("Distribution of headlines per session", y=1.02)
fig.tight_layout()
fig.savefig(str(OUT_DIR / "headlines_per_session.png"), dpi=130, bbox_inches="tight")

print("Seen   :", seen_counts.describe().round(2).to_dict())
print("Unseen :", unseen_counts.describe().round(2).to_dict())
print("Total  :", total_counts.describe().round(2).to_dict())
print(f"Saved {OUT_DIR / 'headlines_per_session.png'}")
