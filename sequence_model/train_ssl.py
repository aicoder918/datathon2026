"""Stage 1 — self-supervised pretraining on bars (0..49) + headlines from
train_seen + public + private splits.

Usage:
    python -m sequence_model.train_ssl --epochs 15 --batch-size 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sequence_model.data_seq import (
    BAR_FEATURE_DIM,
    INPUT_DIM,
    SessionSSLDataset,
    build_session_inputs,
    collate_ssl,
    compute_feature_stats,
    compute_template_impacts,
    load_bars,
    load_headlines,
    save_feature_stats,
)
from sequence_model.model_seq import SequenceModel
from sequence_model.utils_metrics import save_checkpoint, set_seed


def build_all_ssl_inputs(template_impacts) -> dict:
    """Bars 0..49 from train_seen + public + private, keyed by a prefixed
    session id to avoid cross-split collisions. Uses the train-only
    template impact lookup for the headline channel."""
    out = {}
    prefix = {"train_seen": "T", "public": "U", "private": "V"}
    hl_key = {"train_seen": "train", "public": "public", "private": "private"}
    for split in ("train_seen", "public", "private"):
        bars = load_bars(split)
        hls = load_headlines(hl_key[split])
        inp = build_session_inputs(bars, hls, template_impacts)
        for s, v in inp.items():
            out[f"{prefix[split]}{int(s)}"] = v
        print(f"  {split}: {len(inp)} sessions")
    print(f"  total SSL sessions: {len(out)}")
    return out


def ssl_step(model, batch, device, lambda_sign: float = 1.0,
             lambda_recon: float = 1.0, warmup_bars: int = 10):
    """SSL loss = sign-BCE(next-return) + MSE(recon on standardized bars).

    Both objectives ignore loss contributions from the first `warmup_bars`
    predictions, because at those early positions the forward GRU has no
    meaningful context and the only rational prediction is the global mean.
    """
    x = batch["x"].to(device)
    sign_target = batch["sign_target"].to(device)   # (B, T), raw sign
    recon_target = batch["recon_target"].to(device)  # (B, T, bar_dim), standardized
    mask = batch["mask"].to(device)                  # (B, T)

    next_ret_logits, recon = model.forward_ssl(x)

    # --- next-return: BCE on sign(ret_{t+1}) from h_t for t = 0..T-2 ---
    logits = next_ret_logits[:, :-1]     # (B, T-1)
    true = sign_target[:, 1:]            # (B, T-1)
    T_pred = logits.shape[1]
    warmup_pred = (torch.arange(T_pred, device=device) >= warmup_bars).float()
    per_pos_bce = F.binary_cross_entropy_with_logits(logits, true, reduction="none")
    denom_bce = (warmup_pred.sum() * logits.shape[0]).clamp(min=1.0)
    loss_sign = (per_pos_bce * warmup_pred).sum() / denom_bce

    # --- recon: MSE on standardized bars, masked AND past warmup ---
    per_step_err = ((recon - recon_target) ** 2).mean(dim=-1)   # (B, T)
    T_full = per_step_err.shape[1]
    warmup_full = (torch.arange(T_full, device=device) >= warmup_bars).float()
    combined = mask * warmup_full     # broadcast (B, T) * (T,)
    loss_recon = (per_step_err * combined).sum() / combined.sum().clamp(min=1.0)

    loss = lambda_sign * loss_sign + lambda_recon * loss_recon
    return loss, {"loss_sign": float(loss_sign.item()),
                  "loss_recon": float(loss_recon.item())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--warmup-bars", type=int, default=10,
                        help="zero loss contribution from first N bars")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-sign", type=float, default=1.0,
                        help="weight on next-return sign-BCE head")
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=str, default="sequence_model/ckpt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing template impacts from train data...")
    bars_full_train = load_bars("train_full")
    hls_train = load_headlines("train")
    template_impacts, impact_counts = compute_template_impacts(
        bars_full_train, hls_train
    )
    print(f"  impacts: min={template_impacts.min():+.5f} "
          f"max={template_impacts.max():+.5f} "
          f"mean={template_impacts.mean():+.5f} "
          f"(n non-zero templates: {int((impact_counts > 0).sum())})")

    print("Loading SSL data (train_seen + public + private)...")
    session_inputs = build_all_ssl_inputs(template_impacts)

    print("Computing per-channel feature stats on SSL corpus...")
    feature_mean, feature_std = compute_feature_stats(session_inputs)
    stats_path = Path(args.ckpt_dir) / "feature_stats.npz"
    save_feature_stats(stats_path, feature_mean, feature_std)
    print(f"  mean: {np.array2string(feature_mean, precision=4)}")
    print(f"  std:  {np.array2string(feature_std, precision=4)}")
    print(f"  saved -> {stats_path}")

    dataset = SessionSSLDataset(session_inputs, feature_mean, feature_std,
                                mask_prob=args.mask_prob)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_ssl, num_workers=0)

    model = SequenceModel(
        input_dim=INPUT_DIM, bar_dim=BAR_FEATURE_DIM,
        hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    ckpt_path = Path(args.ckpt_dir) / "ssl_encoder.pt"
    print(f"Device={device}, sessions={len(dataset)}, batches/epoch={len(loader)}")

    epoch_bar = tqdm(range(args.epochs), desc="SSL", unit="ep")
    for ep in epoch_bar:
        model.train()
        ep_loss = ep_sign = ep_recon = 0.0
        n_batches = 0
        batch_bar = tqdm(loader, desc=f"ep {ep+1}/{args.epochs}",
                         leave=False, unit="b")
        for batch in batch_bar:
            optim.zero_grad()
            loss, parts = ssl_step(model, batch, device,
                                   lambda_sign=args.lambda_sign,
                                   lambda_recon=args.lambda_recon,
                                   warmup_bars=args.warmup_bars)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            ep_loss += float(loss.item())
            ep_sign += parts["loss_sign"]
            ep_recon += parts["loss_recon"]
            n_batches += 1
            batch_bar.set_postfix(loss=f"{ep_loss/n_batches:.5f}",
                                  sign=f"{ep_sign/n_batches:.5f}",
                                  rec=f"{ep_recon/n_batches:.5f}")
        batch_bar.close()
        epoch_bar.set_postfix(loss=f"{ep_loss/n_batches:.5f}")
        tqdm.write(f"  Epoch {ep+1:02d}/{args.epochs}: "
                   f"loss={ep_loss/n_batches:.5f} "
                   f"(sign_bce={ep_sign/n_batches:.5f}, "
                   f"recon={ep_recon/n_batches:.5f})")

    meta = {
        "args": vars(args),
        "input_dim": INPUT_DIM,
        "bar_dim": BAR_FEATURE_DIM,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "template_impacts": template_impacts.tolist(),
    }
    save_checkpoint(ckpt_path, model, optimizer=None, meta=meta)
    print(f"Saved SSL checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
