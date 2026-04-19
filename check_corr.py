import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/Users/mgershman/Desktop/datathon/datathon2026/submissions")

anchor_path = ROOT / "submission_best_plus_ridge_top10_935_065.csv"
anchor = pd.read_csv(anchor_path)["target_position"].to_numpy(dtype=float)

candidates = [
    ROOT / "chatgpt" / "ridge_all.csv",
    ROOT / "chatgpt" / "ridge_top10.csv",
    ROOT / "chatgpt" / "ridge_robust_q01.csv",
    ROOT / "chatgpt" / "ridge_robust_q02.csv",
    ROOT / "catboost_bars_depth6.csv",
    ROOT / "catboost_bars_seed20.csv",
    ROOT / "catboost_bars_seed5_decay.csv",
]

print("Pearson correlations against the new 2.90305 anchor:")
print("-" * 60)
for path in candidates:
    if not path.exists():
        print(f"[MISSING] {path.name}")
        continue
    pred = pd.read_csv(path)["target_position"].to_numpy(dtype=float)
    corr = np.corrcoef(anchor, pred)[0, 1]
    print(f"{corr:.4f} | {path.name}")
