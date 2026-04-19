from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-weight", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    left = pd.read_csv(args.left).sort_values("session").reset_index(drop=True)
    right = pd.read_csv(args.right).sort_values("session").reset_index(drop=True)
    if not left["session"].equals(right["session"]):
        raise ValueError("session columns do not align")

    w = float(args.left_weight)
    out = left.copy()
    out["target_position"] = (
        w * left["target_position"].to_numpy(dtype=float)
        + (1.0 - w) * right["target_position"].to_numpy(dtype=float)
    )

    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(out["target_position"].describe().to_string())


if __name__ == "__main__":
    main()