"""Visualize how headline templates + relevance predict forward volatility.

Uses headline_features.csv (labeled with fwd_log_vol_k5) to answer:
  1. Which templates predict higher subsequent volatility?
  2. Does it matter whether the headline is about the session's own stock
     (ThisCompany=1) vs. an unrelated company (ThisCompany=0)?
  3. How does that impact evolve across the session timeline (bar_ix)?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "headline_features.csv",
                 usecols=["template_index", "template_text", "bar_ix",
                          "ThisCompany", "ThisSector", "fwd_log_vol_k5",
                          "company", "sector", "region"])
df = df.dropna(subset=["fwd_log_vol_k5"])
print(f"Loaded {len(df):,} labeled headlines, {df['template_index'].nunique()} templates")

BASELINE = df["fwd_log_vol_k5"].mean()


# ---------- panel 1: templates ranked by relative forward vol, split by relevance ----------
def template_stats(sub: pd.DataFrame) -> pd.DataFrame:
    g = sub.groupby("template_index")["fwd_log_vol_k5"].agg(["mean", "count", "std"])
    g["sem"] = g["std"] / np.sqrt(g["count"])
    g["rel"] = g["mean"] / BASELINE - 1  # % above/below global baseline
    return g


own = template_stats(df[df["ThisCompany"] == 1])
other = template_stats(df[df["ThisCompany"] == 0])

# only templates that appear both as own-company and other-company with enough samples
MIN_N = 50
common = own.index.intersection(other.index)
common = [t for t in common if own.loc[t, "count"] >= MIN_N and other.loc[t, "count"] >= MIN_N]

# template labels from the df
tpl_text = df.drop_duplicates("template_index").set_index("template_index")["template_text"]
own_c, other_c = own.loc[common], other.loc[common]

# Panel A: show only the TOP-N templates by relevance amplification (own − other)
# so the story — "own-stock headlines move vol more" — is legible.
gap_all = (own_c["mean"] - other_c["mean"]).sort_values(ascending=False)
TOP_N = 18
top_tids = gap_all.head(TOP_N).index.tolist()
# order panel-A rows by own-company vol so the top row is the loudest template
order = own_c.loc[top_tids, "mean"].sort_values(ascending=True).index.tolist()
ownA, otherA = own_c.loc[order], other_c.loc[order]


# ---------- figure (compacted: dropped redundant all-templates bar chart) ----------
fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1.0], hspace=0.35)

# panel A: horizontal dot plot, own vs other
axA = fig.add_subplot(gs[0, 0])
y = np.arange(len(order))
axA.hlines(y, otherA["mean"], ownA["mean"],
           color="#bbbbbb", linewidth=2.0, zorder=1, alpha=0.7)
axA.scatter(otherA["mean"], y, s=90, color="#888888",
            label="same template, unrelated company", zorder=2,
            edgecolor="black", linewidth=0.4)
axA.scatter(ownA["mean"], y, s=110,
            color=["#d62728" if m > BASELINE else "#2ca02c" for m in ownA["mean"]],
            label="same template, about the traded stock", zorder=3,
            edgecolor="black", linewidth=0.5)
axA.axvline(BASELINE, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
            label=f"global mean ({BASELINE:.4f})")

labels = [(tpl_text.get(t, str(t))[:72] + "…"
           if len(tpl_text.get(t, str(t))) > 72 else tpl_text.get(t, str(t)))
          for t in order]
axA.set_yticks(y)
axA.set_yticklabels(labels, fontsize=9)
axA.set_xlabel("mean forward 5-bar log volatility", fontsize=10)
axA.set_title(f"Top-{TOP_N} templates where relevance matters most\n"
              "— gap between the gray and red dot is the volatility bump "
              "you get when the headline is about THIS stock —",
              fontsize=11)
axA.legend(loc="lower right", fontsize=9, framealpha=0.95)
axA.grid(axis="x", alpha=0.25)
# annotate counts at the right edge so the reader can see sample sizes
xmax = ownA["mean"].max()
for yi, tid in enumerate(order):
    axA.text(xmax * 1.005, yi,
             f"  n={int(ownA.loc[tid, 'count'])} / {int(otherA.loc[tid, 'count'])}",
             va="center", fontsize=7, color="gray")


gap_all_sorted = gap_all.sort_values()
n_pos = int((gap_all_sorted.values > 0).sum())
axA.text(0.02, 0.97,
         f"{n_pos}/{len(gap_all_sorted)} templates with ≥{MIN_N} samples each\n"
         f"show positive own-stock vol amplification",
         transform=axA.transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6",
                   edgecolor="#b58900", alpha=0.95))


# panel C: forward vol vs bar_ix, split by own vs other
axC = fig.add_subplot(gs[1, 0])
for label, mask, color in [
    ("own stock (ThisCompany=1)", df["ThisCompany"] == 1, "#d62728"),
    ("other stock",               df["ThisCompany"] == 0, "#888888"),
]:
    g = df[mask].groupby("bar_ix")["fwd_log_vol_k5"].agg(["mean", "count", "std"])
    g = g[g["count"] >= 30]
    sem = g["std"] / np.sqrt(g["count"])
    axC.plot(g.index, g["mean"], color=color, linewidth=1.8, label=label)
    axC.fill_between(g.index, g["mean"] - 1.96 * sem, g["mean"] + 1.96 * sem,
                     color=color, alpha=0.15)

axC.axhline(BASELINE, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
            label="global mean")
axC.set_xlabel("bar_ix (position in session)", fontsize=9)
axC.set_ylabel("mean forward 5-bar log vol", fontsize=9)
axC.set_title("Volatility impact across session timeline\n(95% CI bands)", fontsize=10)
axC.legend(fontsize=8, loc="best")
axC.grid(alpha=0.25)


fig.suptitle("Headline → forward volatility: relevance matters more than template choice",
             fontsize=13, y=0.995, fontweight="bold")
out = OUT_DIR / "headline_volatility_relevance.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Wrote {out}")

# quick printed summary so we know the numbers behind the plot
top = (own_c["mean"] - other_c["mean"]).sort_values(ascending=False).head(5)
print("\nTop-5 templates by relevance amplification (own − other):")
for tid, v in top.items():
    print(f"  +{v:+.5f}  {tpl_text.get(tid, tid)[:80]}")
