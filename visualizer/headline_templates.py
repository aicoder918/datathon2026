"""Group headlines by extracted template and rank templates by forward price effect."""
import re
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

# Sector / segment phrases that fill template slots — unify them as <SECTOR>
SECTOR_PHRASES = [
    "wireless connectivity", "cloud infrastructure", "enterprise software",
    "renewable storage", "renewable energy", "automated logistics",
    "precision medicine", "supply chain optimization", "supply chain",
    "data analytics", "advanced manufacturing", "consumer electronics",
    "biotechnology", "fintech", "robotics", "autonomous vehicles",
]
REGIONS = [
    "Southeast Asia", "Asia Pacific", "Latin America", "Central Europe",
    "North America", "Eastern Europe", "Middle East", "Scandinavia",
    "Africa", "South America",
]


def extract_template(text: str) -> str:
    t = text
    # 1. dollar amounts: $5.7B, $250M, $50M, $1.2K -> $X
    t = re.sub(r"\$\d+(?:\.\d+)?\s?[BMK]?", "$X", t)
    # 2. percentages: 18%, 0.5% -> N%
    t = re.sub(r"\d+(?:\.\d+)?\s?%", "N%", t)
    # 3. plain numbers (after the above so we don't double-replace) -> N
    t = re.sub(r"\b\d+(?:\.\d+)?\b", "N", t)
    # 4. company names (1-3 capitalized tokens at the very start)
    t = re.sub(r"^(?:[A-Z][A-Za-z]+\s){1,3}", "<COMPANY> ", t)
    # 5. sector/segment substitutions
    for s in SECTOR_PHRASES:
        t = re.sub(re.escape(s), "<SECTOR>", t, flags=re.IGNORECASE)
    for r in REGIONS:
        t = re.sub(re.escape(r), "<REGION>", t, flags=re.IGNORECASE)
    # 6. collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


hl["template"] = hl["headline"].map(extract_template)
print(f"Distinct templates: {hl['template'].nunique()}  (from {len(hl)} headlines)")

# ---------- compute forward return per headline ----------
close = bars.set_index(["session", "bar_ix"])["close"].sort_index()
max_bar_per_session = bars.groupby("session")["bar_ix"].max()

K = 5  # horizon in bars
hl["c0"] = close.reindex(list(zip(hl["session"], hl["bar_ix"]))).values
end_bars = np.minimum(hl["bar_ix"] + K, hl["session"].map(max_bar_per_session))
hl["c1"] = close.reindex(list(zip(hl["session"], end_bars))).values
hl["fwd_ret"] = hl["c1"] / hl["c0"] - 1

# ---------- aggregate per template ----------
agg = hl.groupby("template")["fwd_ret"].agg(["count", "mean", "std"]).reset_index()
agg["sem"] = agg["std"] / np.sqrt(agg["count"])
agg = agg[agg["count"] >= 30].sort_values("mean")  # need enough samples per template

print(f"\nTemplates with >=30 headlines: {len(agg)}")
print(agg.to_string(index=False))

# ---------- plot: horizontal bar chart of templates ranked by forward return ----------
fig, ax = plt.subplots(figsize=(12, max(6, 0.32 * len(agg))))
y = np.arange(len(agg))
colors_arr = ["#d62728" if m < 0 else "#2ca02c" for m in agg["mean"]]
ax.barh(y, agg["mean"], xerr=1.96 * agg["sem"], color=colors_arr, alpha=0.8, capsize=2)
ax.axvline(0, color="black", linewidth=0.6)

# truncate template strings for the labels
labels = [(t if len(t) <= 80 else t[:79] + "…") for t in agg["template"]]
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(f"mean forward return over +{K} bars  (95% CI)")
ax.set_title(f"Headline templates ranked by subsequent {K}-bar return  (n>=30 each)")

# annotate counts on the right
xmax = (agg["mean"] + 1.96 * agg["sem"]).max()
for yi, n in zip(y, agg["count"]):
    ax.text(xmax * 1.02, yi, f"n={n}", va="center", fontsize=7, color="gray")

fig.tight_layout()
out = OUT_DIR / "headline_template_returns.png"
fig.savefig(str(out), dpi=130, bbox_inches="tight")
print(f"\nSaved {out}")
