"""Stage 2 — supervised fine-tune with GroupKFold(5) on train sessions.

y = close[99] / close[49] - 1 (requires full 100 bars from train_seen + train_unseen).
Encoder is initialized from the SSL checkpoint if present; the target head is
trained from scratch.

Usage:
    python -m sequence_model.train_supervised --epochs 30 --n-folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sequence_model.data_seq import (
    BAR_FEATURE_DIM,
    INPUT_DIM,
    SessionSupervisedDataset,
    build_session_inputs,
    collate_supervised,
    compute_supervised_targets,
    compute_template_impacts,
    load_bars,
    load_feature_stats,
    load_headlines,
)
from sequence_model.model_seq import SequenceModel
from sequence_model.utils_metrics import (
    load_checkpoint,
    save_checkpoint,
    set_seed,
    sharpe,
)


def supervised_step(model, batch, device, target_mean: float, target_std: float,
                    aux_weight: float = 0.1, sharpe_weight: float = 0.0,
                    warmup_bars: int = 10):
    x = batch["x"].to(device)
    y = batch["y"].to(device)            # original scale
    y_std = (y - target_mean) / target_std
    y_hat, next_ret_logits = model.forward_supervised(x)  # logits, standardized target
    loss = F.smooth_l1_loss(y_hat, y_std)
    if sharpe_weight > 0:
        pnl = y_hat * y
        sharpe_term = pnl.mean() / (pnl.std() + 1e-6)
        loss = loss - sharpe_weight * sharpe_term
    if aux_weight > 0:
        # Sign-BCE on next bar's return sign, with warmup mask
        sign_target = batch["sign_target"].to(device)   # (B, T)
        logits = next_ret_logits[:, :-1]
        true = sign_target[:, 1:]
        T_pred = logits.shape[1]
        warmup_mask = (torch.arange(T_pred, device=device) >= warmup_bars).float()
        per_pos_bce = F.binary_cross_entropy_with_logits(logits, true,
                                                          reduction="none")
        denom = (warmup_mask.sum() * logits.shape[0]).clamp(min=1.0)
        loss_aux = (per_pos_bce * warmup_mask).sum() / denom
        loss = loss + aux_weight * loss_aux
    return loss, y_hat


@torch.no_grad()
def evaluate(model, loader, device, target_mean: float, target_std: float):
    """Returns predictions inverted to original target scale, plus original ys."""
    model.eval()
    preds, ys = [], []
    for batch in loader:
        x = batch["x"].to(device)
        y_hat, _ = model.forward_supervised(x)
        y_hat_orig = y_hat.cpu().numpy() * target_std + target_mean
        preds.append(y_hat_orig)
        ys.append(batch["y"].numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])
    return preds, ys


def train_fold(model, tr_loader, va_loader, device, args,
               target_mean: float, target_std: float,
               fold_idx: int = 0, total_folds: int = 1) -> tuple[float, dict]:
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    best_sharpe = -float("inf")
    best_state = None
    patience = 0
    epoch_bar = tqdm(range(args.epochs),
                     desc=f"fold {fold_idx+1}/{total_folds}", unit="ep")
    for ep in epoch_bar:
        model.train()
        tr_loss = 0.0
        n = 0
        batch_bar = tqdm(tr_loader, desc=f"  ep {ep+1}",
                         leave=False, unit="b")
        for batch in batch_bar:
            optim.zero_grad()
            loss, _ = supervised_step(model, batch, device,
                                      target_mean=target_mean,
                                      target_std=target_std,
                                      aux_weight=args.aux_weight,
                                      sharpe_weight=args.sharpe_weight,
                                      warmup_bars=args.warmup_bars)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            tr_loss += float(loss.item())
            n += 1
            batch_bar.set_postfix(loss=f"{tr_loss/n:.5f}")
        batch_bar.close()
        preds, ys = evaluate(model, va_loader, device, target_mean, target_std)
        val_sharpe = sharpe(preds, ys)
        pred_std = float(preds.std()) if preds.size else 0.0
        epoch_bar.set_postfix(val_sharpe=f"{val_sharpe:.3f}",
                              best=f"{max(best_sharpe, val_sharpe):.3f}",
                              std=f"{pred_std:.4f}")
        tqdm.write(f"    ep {ep+1:02d}: train_loss={tr_loss/max(n,1):.5f}, "
                   f"val_sharpe={val_sharpe:.3f}, pred_std={pred_std:.5f}")
        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                tqdm.write(f"    early stop at ep {ep+1}")
                break
    epoch_bar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_sharpe, {"best_sharpe": best_sharpe}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--sharpe-weight", type=float, default=0.0,
                        help="weight on -batch_sharpe(y_hat, y) loss term")
    parser.add_argument("--standardize-y", dest="standardize_y",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="z-score y per fold using train indices")
    parser.add_argument("--warmup-bars", type=int, default=10,
                        help="zero aux-loss contribution from first N bars")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssl-ckpt", type=str,
                        default="sequence_model/ckpt/ssl_encoder.pt")
    parser.add_argument("--ckpt-dir", type=str, default="sequence_model/ckpt")
    parser.add_argument("--oof-out", type=str,
                        default="sequence_model/oof_seq.parquet")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train data (100-bar sessions)...")
    bars_full = load_bars("train_full")
    hls = load_headlines("train")

    print("Computing template impacts (train-only)...")
    template_impacts, impact_counts = compute_template_impacts(bars_full, hls)
    print(f"  impacts: min={template_impacts.min():+.5f} "
          f"max={template_impacts.max():+.5f} "
          f"mean={template_impacts.mean():+.5f} "
          f"(n non-zero templates: {int((impact_counts > 0).sum())})")

    session_inputs = build_session_inputs(bars_full, hls, template_impacts)
    targets = compute_supervised_targets(bars_full)

    stats_path = Path(args.ckpt_dir) / "feature_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} missing — run train_ssl first (it saves the stats)."
        )
    feature_mean, feature_std = load_feature_stats(stats_path)
    print(f"Loaded feature stats from {stats_path}: "
          f"mean[{feature_mean.min():+.4f},{feature_mean.max():+.4f}], "
          f"std[{feature_std.min():.4f},{feature_std.max():.4f}]")

    dataset = SessionSupervisedDataset(session_inputs, targets,
                                       feature_mean, feature_std)

    sessions = np.array(dataset.sessions, dtype=int)
    ys_all = np.array([targets[int(s)] for s in sessions], dtype=float)
    print(f"  {len(sessions)} train sessions, target mean={ys_all.mean():.5f}, "
          f"std={ys_all.std():.5f}")

    gkf = GroupKFold(n_splits=args.n_folds)
    oof = np.full(len(sessions), np.nan, dtype=float)
    fold_sharpes = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(sessions, sessions, sessions)):
        print(f"Fold {fold}: train={len(tr_idx)}, val={len(va_idx)}")
        tr_loader = DataLoader(Subset(dataset, tr_idx), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_supervised)
        va_loader = DataLoader(Subset(dataset, va_idx), batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_supervised)

        if args.standardize_y:
            tr_y = ys_all[tr_idx]
            target_mean = float(tr_y.mean())
            target_std = float(tr_y.std()) or 1.0
        else:
            target_mean = 0.0
            target_std = 1.0
        print(f"  target stats: mean={target_mean:+.5f}, std={target_std:.5f}"
              f" (standardize={args.standardize_y})")

        model = SequenceModel(
            input_dim=INPUT_DIM, bar_dim=BAR_FEATURE_DIM,
            hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
        ).to(device)

        ssl_ckpt = Path(args.ssl_ckpt)
        if ssl_ckpt.exists():
            load_checkpoint(ssl_ckpt, model, map_location=device,
                            strict=False, exclude_prefixes=("target_head.",))
            print(f"  loaded SSL encoder from {ssl_ckpt}")
        else:
            print(f"  WARNING: no SSL checkpoint at {ssl_ckpt}, training from scratch")

        best_sharpe, _ = train_fold(model, tr_loader, va_loader, device, args,
                                    target_mean=target_mean,
                                    target_std=target_std,
                                    fold_idx=fold, total_folds=args.n_folds)
        fold_sharpes.append(best_sharpe)

        preds_fold, _ = evaluate(model, va_loader, device, target_mean, target_std)
        oof[va_idx] = preds_fold
        save_checkpoint(ckpt_dir / f"fold_{fold}.pt", model,
                        meta={"fold": fold, "best_sharpe": best_sharpe,
                              "hidden": args.hidden, "n_layers": args.n_layers,
                              "target_mean": target_mean,
                              "target_std": target_std,
                              "template_impacts": template_impacts.tolist()})
        print(f"  fold {fold} best_sharpe={best_sharpe:.3f}")

    oof_sharpe = sharpe(oof, ys_all)
    print(f"\nOOF Sharpe: {oof_sharpe:.3f}")
    print(f"Mean fold Sharpe: {np.mean(fold_sharpes):.3f} "
          f"+/- {np.std(fold_sharpes):.3f}")
    print(f"OOF pred std: {float(np.nanstd(oof)):.5f} "
          f"(should be > 0, else collapse)")

    oof_df = pd.DataFrame({"session": sessions, "oof_pred": oof, "target": ys_all})
    Path(args.oof_out).parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(args.oof_out, index=False)
    print(f"Saved OOF -> {args.oof_out}")


if __name__ == "__main__":
    main()
