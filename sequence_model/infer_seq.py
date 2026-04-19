"""Stage 3 — inference on public + private, optional position shaping, write
submission CSV.

Usage:
    python -m sequence_model.infer_seq --out submission_seq.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sequence_model.data_seq import (
    BAR_FEATURE_DIM,
    INPUT_DIM,
    SessionSupervisedDataset,
    build_session_inputs,
    collate_supervised,
    compute_template_impacts,
    load_bars,
    load_feature_stats,
    load_headlines,
)
from sequence_model.model_seq import SequenceModel
from sequence_model.utils_metrics import load_checkpoint


@torch.no_grad()
def predict_split(split: str, models, fold_stats, device, template_impacts,
                  feature_mean, feature_std, batch_size: int = 64):
    """fold_stats: list of (target_mean, target_std) per fold; inverts each
    fold's standardized prediction back to the original target scale before
    averaging across folds. Inputs are z-scored with (feature_mean, feature_std)
    inside the dataset (must match the stats used at training time)."""
    bars = load_bars(split)
    hls = load_headlines(split)
    session_inputs = build_session_inputs(bars, hls, template_impacts)
    dummy_targets = {int(s): 0.0 for s in session_inputs}
    ds = SessionSupervisedDataset(session_inputs, dummy_targets,
                                  feature_mean, feature_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_supervised)

    preds, sessions = [], []
    for batch in tqdm(loader, desc=f"predict {split}", unit="b"):
        x = batch["x"].to(device)
        fold_outputs = []
        for m, (mu, sigma) in zip(models, fold_stats):
            y_hat_std = m.forward_supervised(x)[0].cpu().numpy()
            fold_outputs.append(y_hat_std * sigma + mu)  # invert to original scale
        preds.append(np.stack(fold_outputs, axis=0).mean(axis=0))
        sessions.append(batch["session"].numpy())
    return np.concatenate(sessions), np.concatenate(preds)


def compute_vol_proxy(bars_df: pd.DataFrame) -> dict:
    """std(returns) over bars 0..49 per session, used as inverse-vol scaler."""
    out = {}
    for s, g in bars_df.groupby("session"):
        g = g.sort_values("bar_ix")
        c = g["close"].to_numpy(dtype=float)
        ret = np.zeros_like(c)
        if len(c) > 1:
            ret[1:] = c[1:] / (c[:-1] + 1e-8) - 1.0
        out[int(s)] = float(np.std(ret) + 1e-6)
    return out


def shape_positions(pred, vol, threshold_q: float = 0.35,
                    blend_alpha: float = 0.5, short_floor: float = 0.3):
    pred = np.asarray(pred, dtype=float)
    vol = np.asarray(vol, dtype=float)
    abs_pred = np.abs(pred)
    cutoff = float(np.quantile(abs_pred, threshold_q)) if threshold_q > 0 else 0.0
    pos = pred.copy()
    pos[abs_pred < cutoff] = 0.0
    pos = pos / np.maximum(vol, 1e-6)
    m = float(np.mean(np.abs(pos)))
    if m > 1e-12:
        pos = pos / m
    pos = blend_alpha * pos + (1.0 - blend_alpha) * 1.0
    pos = np.maximum(pos, short_floor)
    return pos, cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="sequence_model/ckpt")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold-q", type=float, default=0.35)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--short-floor", type=float, default=0.3)
    parser.add_argument("--no-shape", action="store_true")
    parser.add_argument("--out", type=str, default="submission_seq.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)

    fold_paths = sorted(ckpt_dir.glob("fold_*.pt"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.pt found in {ckpt_dir}")
    print(f"Loading {len(fold_paths)} fold checkpoints")

    models = []
    fold_stats = []  # list of (target_mean, target_std) per fold
    for p in fold_paths:
        m = SequenceModel(
            input_dim=INPUT_DIM, bar_dim=BAR_FEATURE_DIM,
            hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
        ).to(device)
        meta = load_checkpoint(p, m, map_location=device)
        m.eval()
        models.append(m)
        mu = float(meta.get("target_mean", 0.0))
        sigma = float(meta.get("target_std", 1.0)) or 1.0
        fold_stats.append((mu, sigma))
        print(f"  {p.name}: target_mean={mu:+.5f}, target_std={sigma:.5f}")

    stats_path = ckpt_dir / "feature_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} missing — re-run train_ssl to regenerate."
        )
    feature_mean, feature_std = load_feature_stats(stats_path)
    print(f"Loaded feature stats from {stats_path}")

    print("Computing template impacts (train-only)...")
    bars_full = load_bars("train_full")
    hls_train = load_headlines("train")
    template_impacts, impact_counts = compute_template_impacts(bars_full, hls_train)
    print(f"  impacts: min={template_impacts.min():+.5f} "
          f"max={template_impacts.max():+.5f} "
          f"mean={template_impacts.mean():+.5f} "
          f"(n non-zero templates: {int((impact_counts > 0).sum())})")

    parts = []
    for split in ("public", "private"):
        print(f"Predicting {split}...")
        sessions, preds = predict_split(split, models, fold_stats, device,
                                        template_impacts,
                                        feature_mean, feature_std)
        print(f"  raw preds: mean={preds.mean():.6f}, std={preds.std():.6f}")
        if args.no_shape:
            positions = preds
            cutoff = 0.0
        else:
            bars = load_bars(split)
            vol_map = compute_vol_proxy(bars)
            vol = np.array([vol_map[int(s)] for s in sessions], dtype=float)
            positions, cutoff = shape_positions(
                preds, vol,
                threshold_q=args.threshold_q,
                blend_alpha=args.blend_alpha,
                short_floor=args.short_floor,
            )
        print(f"  shaped: std={positions.std():.6f}, "
              f"frac positive={float((positions > 0).mean()):.1%}, "
              f"cutoff={cutoff:.6g}")
        parts.append(pd.DataFrame({
            "session": sessions.astype(int),
            "target_position": positions,
        }))

    submission = (pd.concat(parts, ignore_index=True)
                    .sort_values("session")
                    .reset_index(drop=True))
    assert submission["session"].is_unique, "duplicate sessions"
    assert submission["target_position"].notna().all(), "NaN in positions"
    submission.to_csv(args.out, index=False)
    print(f"\nSaved submission ({len(submission)} rows) -> {args.out}")


if __name__ == "__main__":
    main()
