"""Refresh the FinBERT cache so train and test headline features use full coverage.

The current cache at data/headlines_finbert_sentiment.parquet covers train
headlines but misses many public/private test headlines. That silently pushes
test-time sentiment aggregates toward zero.

Usage:
    ../.venv/bin/python models/update_sentiment_cache.py
    ../.venv/bin/python models/update_sentiment_cache.py --limit 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_PATH = DATA_DIR / "headlines_finbert_sentiment.parquet"

HEADLINE_FILES = [
    "headlines_seen_train.parquet",
    "headlines_unseen_train.parquet",
    "headlines_seen_public_test.parquet",
    "headlines_seen_private_test.parquet",
]


def score_finbert(texts: list[str], batch_size: int) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "ProsusAI/finbert"
    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()
    id2label = model.config.id2label

    labels: list[str] = []
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            idx = probs.argmax(axis=-1)
            labels.extend(id2label[int(i)] for i in idx)
            scores.extend(float(probs[row, int(i)]) for row, i in enumerate(idx))
            done = start + len(chunk)
            if done == len(texts) or (done // batch_size) % 10 == 0:
                print(f"  scored {done}/{len(texts)}")
    return pd.DataFrame({"headline": texts, "label": labels, "score": scores})


def load_all_unique_headlines() -> pd.Index:
    parts = []
    for rel in HEADLINE_FILES:
        path = DATA_DIR / rel
        if not path.exists():
            continue
        parts.append(pd.read_parquet(path, columns=["headline"])["headline"])
    if not parts:
        raise FileNotFoundError("No headline parquet files found.")
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def summarize_coverage(headlines: pd.Index, cache: pd.DataFrame, label: str) -> None:
    covered = headlines.isin(set(cache["headline"])).mean()
    print(
        f"{label}: covered={covered:.4f} "
        f"({int(round(covered * len(headlines)))}/{len(headlines)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N missing headlines for a smoke test.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="FinBERT batch size.",
    )
    args = ap.parse_args()

    headlines = load_all_unique_headlines()
    if CACHE_PATH.exists():
        cache = pd.read_parquet(CACHE_PATH)
        cache = cache.drop_duplicates("headline", keep="last")
    else:
        cache = pd.DataFrame(columns=["headline", "label", "score"])

    summarize_coverage(headlines, cache, "before")
    missing = headlines[~headlines.isin(set(cache["headline"]))].tolist()
    if args.limit is not None:
        missing = missing[:args.limit]

    print(f"unique headlines={len(headlines)} cached={len(cache)} missing={len(missing)}")
    if not missing:
        print("Cache already covers all requested headlines.")
        return

    scored = score_finbert(missing, batch_size=args.batch_size)
    cache = pd.concat([cache, scored], ignore_index=True)
    cache = cache.drop_duplicates("headline", keep="last").sort_values("headline").reset_index(drop=True)
    cache.to_parquet(CACHE_PATH, index=False)
    print(f"Saved {len(cache)} rows to {CACHE_PATH}")
    summarize_coverage(headlines, cache, "after")


if __name__ == "__main__":
    main()
