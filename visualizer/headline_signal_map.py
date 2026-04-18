"""Second volatility-insights plot: four orthogonal signals the model can use.

1. Relevance 2x2: ThisCompany × ThisSector — does sector-match help when the
   company doesn't match?
2. Dollar-magnitude bucket: do bigger $ headlines predict bigger vol?
3. Session-level density: more own-company headlines per session → higher vol?
4. Event-type clusters: group the 73 templates by action verb and rank clusters
   by vol impact (compresses template_text into ~8 interpretable buckets).
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "headline_features.csv",
                 usecols=["template_index", "template_text", "bar_ix", "session",
                          "ThisCompany", "ThisSector", "fwd_log_vol_k5",
                          "dollar", "percentage", "company",
                          "session_top_company_mean_vol"])
df = df.dropna(subset=["fwd_log_vol_k5"])
BASELINE = df["fwd_log_vol_k5"].mean()


# ---------- panel 1: ThisCompany × ThisSector heatmap ----------
grid = df.groupby(["ThisCompany", "ThisSector"])["fwd_log_vol_k5"].agg(["mean", "count"])
M = np.array([[grid.loc[(0, 0), "mean"], grid.loc[(0, 1), "mean"]],
              [grid.loc[(1, 0), "mean"], grid.loc[(1, 1), "mean"]]])
N = np.array([[grid.loc[(0, 0), "count"], grid.loc[(0, 1), "count"]],
              [grid.loc[(1, 0), "count"], grid.loc[(1, 1), "count"]]])


# ---------- panel 2: dollar magnitude → forward vol ----------
def parse_dollar(s):
    if not isinstance(s, str):
        return np.nan
    m = re.match(r"\$([\d.]+)\s?([KMBT]?)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    mult = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)]
    return v * mult


df["dollar_val"] = df["dollar"].map(parse_dollar)
d = df.dropna(subset=["dollar_val"]).copy()
bins = [0, 1e8, 3e8, 6e8, 1e9, 5e9, np.inf]
labels = ["<$100M", "$100–300M", "$300–600M", "$600M–$1B", "$1–5B", "$5B+"]
d["bucket"] = pd.cut(d["dollar_val"], bins=bins, labels=labels)
# split by own vs other
dollar_stats = d.groupby(["bucket", "ThisCompany"])["fwd_log_vol_k5"].agg(["mean", "count", "std"]).reset_index()
dollar_stats["sem"] = dollar_stats["std"] / np.sqrt(dollar_stats["count"])


# ---------- panel 3: session-level headline count vs realized vol proxy ----------
# per-session: how many ThisCompany=1 headlines, and the session's mean realized vol
# (session_top_company_mean_vol is constant within a session)
sess = df.groupby("session").agg(
    n_own=("ThisCompany", "sum"),
    n_total=("ThisCompany", "count"),
    sess_vol=("session_top_company_mean_vol", "first"),
).reset_index()
sess["own_share"] = sess["n_own"] / sess["n_total"]
# bin by n_own
sess["own_bucket"] = pd.cut(sess["n_own"], bins=[0.5, 1.5, 2.5, 3.5, 5.5, 100],
                             labels=["1", "2", "3", "4–5", "6+"])
sess_stats = sess.groupby("own_bucket", observed=True)["sess_vol"].agg(
    ["mean", "median", "count", "std"]).reset_index()
sess_stats["sem"] = sess_stats["std"] / np.sqrt(sess_stats["count"])


# ---------- panel 4: event-type clusters ----------
# Map each template_text to a coarse event category based on action verb / keyword.
CATEGORIES = [
    ("earnings beat/miss", r"earnings|beats|misses|guidance|margin|quarter|profit|revenue"),
    ("contract/deal",      r"contract|deal|partnership|partner|agreement"),
    ("expansion",          r"expands|opens|new office|launch|enters|facility"),
    ("M&A / capital",      r"acquisition|acquires|merger|buyback|capital|expenditure|stake|investment"),
    ("product / tech",     r"product|patent|prototype|technology|rollout|unveil|platform"),
    ("leadership",         r"CEO|CFO|CTO|executive|appoints|resign|chief"),
    ("regulatory / legal", r"regulatory|investigation|lawsuit|class action|fine|probe|review"),
    ("negative event",     r"recall|loses|decline|warns|cut|disruption|delay|suspend|halt|pressure"),
]


def categorize(text: str) -> str:
    for name, pat in CATEGORIES:
        if re.search(pat, text, re.IGNORECASE):
            return name
    return "other"


tpl_map = df.drop_duplicates("template_index")[["template_index", "template_text"]].copy()
tpl_map["category"] = tpl_map["template_text"].map(categorize)
df = df.merge(tpl_map[["template_index", "category"]], on="template_index", how="left")

cat_stats = df[df["ThisCompany"] == 1].groupby("category")["fwd_log_vol_k5"].agg(
    ["mean", "count", "std"]).reset_index()
cat_stats["sem"] = cat_stats["std"] / np.sqrt(cat_stats["count"])
cat_stats = cat_stats[cat_stats["count"] >= 30].sort_values("mean")


# ---------- draw ----------
fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28)

# Panel 1 — 2x2 relevance heatmap
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(M, cmap="Reds", aspect="auto")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["ThisSector=0", "ThisSector=1"])
ax1.set_yticks([0, 1]); ax1.set_yticklabels(["ThisCompany=0", "ThisCompany=1"])
for i in range(2):
    for j in range(2):
        lift = (M[i, j] / BASELINE - 1) * 100
        ax1.text(j, i, f"{M[i, j]:.4f}\n({lift:+.0f}% vs mean)\nn={N[i, j]:,}",
                 ha="center", va="center",
                 color="white" if M[i, j] > M.mean() else "black", fontsize=10,
                 fontweight="bold")
ax1.set_title("Relevance 2×2: own-sector alone outperforms own-company alone\n"
              "(mean forward 5-bar log vol per cell)", fontsize=10)
plt.colorbar(im, ax=ax1, shrink=0.7)


# Panel 2 — dollar magnitude
ax2 = fig.add_subplot(gs[0, 1])
for tc, color, label in [(1, "#d62728", "about THIS stock"), (0, "#888888", "other stock")]:
    sub = dollar_stats[dollar_stats["ThisCompany"] == tc]
    ax2.errorbar(range(len(sub)), sub["mean"], yerr=1.96 * sub["sem"],
                 marker="o", markersize=8, color=color, label=label,
                 linewidth=1.8, capsize=3)
ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=20, ha="right")
ax2.axhline(BASELINE, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="global mean")
ax2.set_ylabel("mean forward 5-bar log vol")
ax2.set_title("Does $ size of the deal predict bigger vol moves?\n"
              "(flat line ⇒ dollar magnitude adds little beyond the template itself)",
              fontsize=10)
ax2.legend(fontsize=8, loc="best")
ax2.grid(alpha=0.25)


# Panel 3 — session-level
ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(len(sess_stats))
ax3.bar(x, sess_stats["mean"], yerr=1.96 * sess_stats["sem"],
        color="#1f77b4", alpha=0.8, capsize=3, edgecolor="black", linewidth=0.4,
        label="mean (with 95% CI)")
ax3.plot(x, sess_stats["median"], "o-", color="#ff9900", markersize=7,
         linewidth=1.6, label="median", zorder=5)
ax3.set_xticks(x); ax3.set_xticklabels(sess_stats["own_bucket"].astype(str))
ax3.set_xlabel("# own-company headlines in session")
ax3.set_ylabel("session volatility")
ax3.set_title("Counterintuitive: sparse-coverage sessions are the loudest\n"
              "1 own-company headline ⇒ ~18% higher vol than 2+",
              fontsize=10)
for xi, (m, n) in enumerate(zip(sess_stats["mean"], sess_stats["count"])):
    ax3.text(xi, m * 1.01, f"n={n}", ha="center", fontsize=8, color="#333")
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(axis="y", alpha=0.25)


# Panel 4 — event-type clusters
ax4 = fig.add_subplot(gs[1, 1])
y = np.arange(len(cat_stats))
colors = ["#d62728" if m > BASELINE else "#2ca02c" for m in cat_stats["mean"]]
ax4.barh(y, cat_stats["mean"], xerr=1.96 * cat_stats["sem"],
         color=colors, alpha=0.85, capsize=3, edgecolor="black", linewidth=0.4)
ax4.axvline(BASELINE, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
            label="global mean")
ax4.set_yticks(y); ax4.set_yticklabels(cat_stats["category"], fontsize=9)
ax4.set_xlabel("mean forward 5-bar log vol (own-stock headlines only)")
ax4.set_title("Templates compressed into 8 event categories\n"
              "(ranked by vol impact — negative events aren't the loudest)",
              fontsize=10)
for yi, (m, n) in enumerate(zip(cat_stats["mean"], cat_stats["count"])):
    ax4.text(m + 1e-5, yi, f"  n={n}", va="center", fontsize=8, color="#555")
ax4.legend(fontsize=8, loc="lower right")
ax4.grid(axis="x", alpha=0.25)


fig.suptitle("Headline signal map — four orthogonal axes for the vol model",
             fontsize=13, y=0.995, fontweight="bold")
out = OUT_DIR / "headline_signal_map.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Wrote {out}")

# ---------- printed summary ----------
print("\n=== 2x2 relevance cells (mean fwd vol) ===")
for i, tc in enumerate([0, 1]):
    for j, ts in enumerate([0, 1]):
        print(f"  ThisCompany={tc}, ThisSector={ts}: "
              f"{M[i, j]:.5f}  ({(M[i, j] / BASELINE - 1) * 100:+.1f}% vs global) "
              f"n={N[i, j]:,}")

print("\n=== session-level (own-company count → session vol) ===")
print(sess_stats[["own_bucket", "mean", "count"]].to_string(index=False))

print("\n=== event categories (own-stock only) ===")
print(cat_stats[["category", "mean", "count"]].to_string(index=False))
