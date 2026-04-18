"""Rule-based headline -> template_index parser.

Mirrors the normalization logic that produced `headline_features.csv`'s
`template_index` column. Applying it ourselves means we can assign canonical
template ids to ANY headline (including private-test ones not in the public CSV),
consistent with how the organizers labeled train + public test.

Pipeline: normalize the headline (amounts -> $X, percents -> N%, numbers -> N,
leading capitalized words -> <COMPANY>, known sector/region/partner phrases ->
their placeholders), then look up the normalized string in a
template_text -> index table built from headline_features.csv.

Exports `parse_template_index(headline) -> int` (returns -1 if unmatched).
"""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent

SECTOR_PHRASES = [
    "wireless connectivity", "cloud infrastructure", "enterprise software",
    "renewable storage", "renewable energy", "automated logistics",
    "precision medicine", "supply chain optimization", "supply chain",
    "data analytics", "advanced manufacturing", "consumer electronics",
    "biotechnology", "fintech", "robotics", "autonomous vehicles",
    "precision manufacturing", "process automation", "digital payments",
    "precision manufacturing systems", "process automation systems",
    "digital payments systems",
]
REGIONS = [
    "Southeast Asia", "Asia Pacific", "Latin America", "Central Europe",
    "North America", "Eastern Europe", "Middle East", "Scandinavia",
    "Africa", "South America", "Europe",
]
PARTNER_PHRASES = [
    "a multinational manufacturer", "an international consortium",
    "a top-tier research institute", "a leading cloud platform",
    "a major logistics provider", "a global retailer",
    "a national infrastructure agency",
]

_SECTOR_SORTED = sorted(SECTOR_PHRASES, key=len, reverse=True)
_REGION_SORTED = sorted(REGIONS, key=len, reverse=True)
_PARTNER_SORTED = sorted(PARTNER_PHRASES, key=len, reverse=True)


def _normalize(text: str) -> str:
    t = str(text)
    t = re.sub(r"\$\d+(?:\.\d+)?\s?[BMK]?", "$X", t)
    t = re.sub(r"\d+(?:\.\d+)?\s?%", "N%", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\b", "N", t)
    t = re.sub(r"^(?:[A-Z][A-Za-z]+\s){1,3}", "<COMPANY> ", t)
    for p in _PARTNER_SORTED:
        t = re.sub(re.escape(p), "<PARTNER>", t, flags=re.IGNORECASE)
    for s in _SECTOR_SORTED:
        t = re.sub(re.escape(s), "<SECTOR>", t, flags=re.IGNORECASE)
    for r in _REGION_SORTED:
        t = re.sub(re.escape(r), "<REGION>", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Build template_text -> template_index from the canonical CSV.
_hf = pd.read_csv(MODELS_DIR / "headline_features.csv",
                  usecols=["template_index", "template_text"]).drop_duplicates()
_hf = _hf[_hf["template_index"] >= 0]
TEMPLATE_TO_INDEX: dict[str, int] = dict(
    zip(_hf["template_text"].astype(str), _hf["template_index"].astype(int))
)


@lru_cache(maxsize=None)
def parse_template_index(headline: str) -> int:
    """Normalize a headline and return its canonical template_index (-1 if unmatched)."""
    return TEMPLATE_TO_INDEX.get(_normalize(headline), -1)


if __name__ == "__main__":
    # Sanity check: agreement with headline_features_public.csv.
    pub = pd.read_csv(MODELS_DIR / "headline_features_public.csv",
                      usecols=["title", "template_index"])
    pub["parsed"] = pub["title"].map(parse_template_index)
    agree = (pub["parsed"] == pub["template_index"]).mean()
    both_matched = ((pub["parsed"] >= 0) & (pub["template_index"] >= 0)).mean()
    print(f"templates in table: {len(TEMPLATE_TO_INDEX)}")
    print(f"agreement with public CSV: {agree:.4f}")
    print(f"both matched (>=0):        {both_matched:.4f}")
    print(f"parser -1 rate:  {(pub['parsed'] == -1).mean():.4f}")
    print(f"csv -1 rate:     {(pub['template_index'] == -1).mean():.4f}")
