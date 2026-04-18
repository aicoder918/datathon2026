import pandas as pd
import textwrap
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

seen = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
unseen = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
hl_seen = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
hl_unseen = pd.read_parquet(DATA_DIR / "headlines_unseen_train.parquet")

full = pd.concat([seen.assign(part="seen"), unseen.assign(part="unseen")], ignore_index=True)
full = full.sort_values(["session", "bar_ix"]).reset_index(drop=True)
headlines = pd.concat([hl_seen.assign(part="seen"), hl_unseen.assign(part="unseen")],
                      ignore_index=True).sort_values(["session", "bar_ix"]).reset_index(drop=True)

split_ix = int(seen["bar_ix"].max())  # last seen bar index (49)


CACHE_PATH = DATA_DIR / "headlines_finbert_sentiment.parquet"
LABEL_TO_INT = {"positive": 1, "negative": -1, "neutral": 0}


def score_finbert(texts: list[str]) -> pd.DataFrame:
    """Run FinBERT (ProsusAI/finbert) over headlines. Returns a df with label + score."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = "ProsusAI/finbert"
    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    id2label = model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}

    batch = 64
    all_labels, all_scores = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            idx = probs.argmax(axis=-1)
            all_labels.extend(id2label[int(j)] for j in idx)
            all_scores.extend(float(probs[k, int(idx[k])]) for k in range(len(chunk)))
            if (i // batch) % 10 == 0:
                print(f"  scored {i + len(chunk)}/{len(texts)}")
    return pd.DataFrame({"label": all_labels, "score": all_scores})


# Score (with caching keyed by exact headline text)
unique_headlines = headlines["headline"].drop_duplicates().reset_index(drop=True)
if CACHE_PATH.exists():
    cache = pd.read_parquet(CACHE_PATH)
    missing = unique_headlines[~unique_headlines.isin(cache["headline"])].tolist()
else:
    cache = pd.DataFrame(columns=["headline", "label", "score"])
    missing = unique_headlines.tolist()

if missing:
    print(f"Scoring {len(missing)} new headlines with FinBERT (cached: {len(cache)})")
    new_scores = score_finbert(missing)
    new_scores.insert(0, "headline", missing)
    cache = pd.concat([cache, new_scores], ignore_index=True)
    cache.to_parquet(CACHE_PATH, index=False)
    print(f"Cache updated: {CACHE_PATH}")
else:
    print(f"All {len(unique_headlines)} headlines already cached at {CACHE_PATH}")

headlines = headlines.merge(cache, on="headline", how="left")
headlines["sentiment"] = headlines["label"].map(LABEL_TO_INT).astype(int)
N_SESSIONS = 30  # number of sessions selectable in the dropdown
SESSIONS = sorted(full["session"].unique().tolist())[:N_SESSIONS]


def session_traces(session_id):
    s = full[full["session"] == session_id]
    s_seen = s[s["part"] == "seen"]
    s_unseen = s[s["part"] == "unseen"]

    traces = []

    # Candlesticks: seen
    traces.append(go.Candlestick(
        x=s_seen["bar_ix"], open=s_seen["open"], high=s_seen["high"],
        low=s_seen["low"], close=s_seen["close"],
        name="seen",
        increasing_line_color="#1f77b4", decreasing_line_color="#1f77b4",
        increasing_fillcolor="#1f77b4", decreasing_fillcolor="#9ecae1",
        showlegend=True,
    ))
    # Candlesticks: unseen
    traces.append(go.Candlestick(
        x=s_unseen["bar_ix"], open=s_unseen["open"], high=s_unseen["high"],
        low=s_unseen["low"], close=s_unseen["close"],
        name="unseen",
        increasing_line_color="#d62728", decreasing_line_color="#d62728",
        increasing_fillcolor="#d62728", decreasing_fillcolor="#fcae91",
        showlegend=True,
    ))

    # Headline markers — vertical line from top down to the bar's high; hover shows text
    hs = headlines[headlines["session"] == session_id]
    if len(hs):
        hi = float(s["high"].max())
        lo = float(s["low"].min())
        span = hi - lo if hi > lo else 1.0
        top_y = hi + span * 0.10

        bar_high = s.set_index("bar_ix")["high"]

        # One trace per sentiment class so the legend reads green / red / gray.
        sentiment_styles = [
            (1, "positive", "#2ca02c"),
            (-1, "negative", "#d62728"),
            (0, "neutral", "#888888"),
        ]
        for sval, label, color in sentiment_styles:
            sub = hs[hs["sentiment"] == sval]
            if not len(sub):
                continue

            # vertical line segments (visual only; hover handled by marker trace)
            line_x, line_y = [], []
            for row in sub.itertuples(index=False):
                hb = float(bar_high.get(row.bar_ix, hi))
                line_x += [row.bar_ix, row.bar_ix, None]
                line_y += [top_y, hb, None]
            traces.append(go.Scatter(
                x=line_x, y=line_y, mode="lines",
                line=dict(color=color, width=1.2),
                hoverinfo="skip", showlegend=False,
            ))

            marker_x = sub["bar_ix"].tolist()
            marker_y = [top_y] * len(sub)
            wrapped = ["<br>".join(textwrap.wrap(h, 60)) for h in sub["headline"]]
            cd = [(int(b), p, label, float(sc))
                  for b, p, sc in zip(sub["bar_ix"], sub["part"], sub["score"])]
            traces.append(go.Scatter(
                x=marker_x, y=marker_y, mode="markers",
                marker=dict(symbol="triangle-down", size=14, color=color,
                            line=dict(width=1, color="black")),
                text=wrapped,
                customdata=cd,
                hovertemplate=("<b>bar %{customdata[0]} (%{customdata[1]})</b>"
                               " — <i>%{customdata[2]} (%{customdata[3]:.2f})</i>"
                               "<br>%{text}<extra></extra>"),
                name=f"{label} headlines",
                hoverlabel=dict(bgcolor="white", font_size=11),
            ))
    return traces


# Build one figure with all sessions; toggle visibility via dropdown
fig = go.Figure()
trace_session = []  # parallel list: which session each trace belongs to
for sid in SESSIONS:
    ts = session_traces(sid)
    for t in ts:
        fig.add_trace(t)
        trace_session.append(sid)

# Initially show only the first session
default_sid = SESSIONS[0]
for i, sid in enumerate(trace_session):
    fig.data[i].visible = (sid == default_sid)

# Build dropdown buttons
buttons = []
for sid in SESSIONS:
    visibility = [s == sid for s in trace_session]
    buttons.append(dict(
        label=f"Session {sid}",
        method="update",
        args=[{"visible": visibility},
              {"title.text": f"Session {sid} — OHLC + headlines (hover markers)"}],
    ))

fig.update_layout(
    title=f"Session {default_sid} — OHLC + headlines (hover markers)",
    xaxis_title="bar_ix",
    yaxis_title="price",
    xaxis_rangeslider_visible=False,
    height=650,
    updatemenus=[dict(
        active=0, buttons=buttons, x=1.02, xanchor="left", y=1, yanchor="top",
        showactive=True,
    )],
    shapes=[dict(
        type="line", xref="x", yref="paper",
        x0=split_ix + 0.5, x1=split_ix + 0.5, y0=0, y1=1,
        line=dict(color="black", dash="dash", width=1),
    )],
    annotations=[dict(
        x=split_ix + 0.5, y=1.02, xref="x", yref="paper",
        text="seen | unseen", showarrow=False, font=dict(size=10),
    )],
)

out_html = OUT_DIR / "candles_interactive.html"
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Wrote interactive plot: {out_html}")
