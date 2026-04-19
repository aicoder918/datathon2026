"""Convex blend of N submission CSVs: out = sum_i w_i * position_i (same sessions)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--term",
        action="append",
        metavar="PATH:WEIGHT",
        required=True,
        help="Repeatable: submission.csv:0.25 (weights should sum to ~1)",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    terms: list[tuple[Path, float]] = []
    for t in args.term:
        path_s, w_s = t.rsplit(":", 1)
        terms.append((Path(path_s), float(w_s)))

    wsum = sum(w for _, w in terms)
    if abs(wsum - 1.0) > 1e-4:
        raise SystemExit(f"weights sum to {wsum}, expected 1.0")

    base = pd.read_csv(terms[0][0]).sort_values("session").reset_index(drop=True)
    sessions = base["session"]
    acc = np.zeros(len(base), dtype=float)
    for path, w in terms:
        df = pd.read_csv(path).sort_values("session").reset_index(drop=True)
        if not df["session"].equals(sessions):
            raise ValueError(f"session mismatch: {path}")
        acc += w * df["target_position"].to_numpy(dtype=float)

    out = pd.DataFrame({"session": sessions.astype(int), "target_position": acc})
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"wrote {outp}")
    print(out["target_position"].describe().to_string())


if __name__ == "__main__":
    main()
