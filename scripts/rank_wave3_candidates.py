#!/usr/bin/env python3
"""Rank sub_w3_* blends vs LB anchors — pick a short submit shortlist.

Anchors are geometry-identical to public ~2.90306 winners (same three CSV branches).
Candidates that are nearly identical (corr → 1) add little information on LB.
Prefer a spread: anchor itself + candidates most *different* while staying in the mesh
(higher MAD / lower Pearson corr to anchor mean position).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"

ANCHORS = [
    SUB / "sub_base_rt10_ra50_935_055_010.csv",
    SUB / "sub_base_rt10_ra50_935_060_005.csv",
    SUB / "sub_base_rt10_ra50_930_060_010.csv",
    SUB / "sub_base_rt10_ra50_930_055_015.csv",
]


def load(path: Path) -> pd.Series:
    d = pd.read_csv(path).sort_values("session").reset_index(drop=True)
    return d["target_position"].astype(np.float64)


def main() -> None:
    stacks = [load(p).values for p in ANCHORS if p.exists()]
    if not stacks:
        print("Anchor files missing; run wave2 first.")
        return
    ref = np.mean(np.column_stack(stacks), axis=1)

    rows = []
    for p in sorted(SUB.glob("sub_w3_*.csv")):
        y = load(p).values
        if len(y) != len(ref):
            continue
        c = float(np.corrcoef(ref, y)[0, 1])
        mad = float(np.mean(np.abs(y - ref)))
        rows.append((mad, c, p.name))

    rows.sort(reverse=True)
    print("Suggested shortlist (most distributional change vs anchor cloud; still same model stack):")
    for mad, c, name in rows[:8]:
        print(f"  corr={c:.6f}  mad_vs_anchor_mean={mad:.6f}  {name}")
    print("\nAlso submit-control: duplicates of existing LB names if generated:")
    for _, _, name in rows:
        if "935_055_010" in name or "935_060_005" in name or "930_060_010" in name or "930_055_015" in name:
            print(f"  matches prior winner vertex: {name}")


if __name__ == "__main__":
    main()
